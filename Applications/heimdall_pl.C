#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <string>
#include <memory>

#include "hd/parse_command_line.h"
#include "hd/default_params.h"
#include "hd/pipeline.h"
#include "hd/error.h"

// input formats supported
#include "hd/DataSource.h"
#include "hd/SigprocFile.h"
#ifdef HAVE_PSRDADA
#include "hd/PSRDadaRingBuffer.h"
#endif

#include "hd/stopwatch.h"
#include <chrono>
#include <omp.h>

// 定义表示Pipeline任务的结构体
struct PipelineTask {
  hd_pipeline pipeline;             // 已创建的pipeline
  std::vector<hd_byte> filterbank;  // 文件数据缓冲区
  std::string filename;             // 文件名
  size_t nsamps_gulp;               // 一次读取的样本数
  size_t stride;                    // 步长
  size_t nbits;                     // 位深度
  DataSource* data_source;          // 数据源
  
  // 构造函数和析构函数
  PipelineTask() : pipeline(nullptr), data_source(nullptr) {}
  ~PipelineTask() {
    if (pipeline) {
      hd_destroy_pipeline(pipeline);
    }
    if (data_source) {
      delete data_source;
    }
  }
};

// 线程安全的任务队列
class TaskQueue {
private:
  std::queue<std::shared_ptr<PipelineTask>> tasks;
  std::mutex mutex;
  std::condition_variable condition;
  std::atomic<bool> stop_flag{false};

public:
  // 添加任务到队列
  void push(std::shared_ptr<PipelineTask> task) {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.push(task);
    lock.unlock();
    condition.notify_one();
  }

  // 从队列获取任务
  std::shared_ptr<PipelineTask> pop() {
    std::unique_lock<std::mutex> lock(mutex);
    // 当队列为空且未停止时等待
    while (tasks.empty() && !stop_flag) {
      condition.wait(lock);
    }
    // 如果队列为空且已停止，返回nullptr
    if (tasks.empty()) {
      return nullptr;
    }
    std::shared_ptr<PipelineTask> task = tasks.front();
    tasks.pop();
    return task;
  }

  // 停止所有等待线程
  void stop() {
    std::unique_lock<std::mutex> lock(mutex);
    stop_flag = true;
    lock.unlock();
    condition.notify_all();
  }

  // 检查队列是否为空
  bool empty() {
    std::unique_lock<std::mutex> lock(mutex);
    return tasks.empty();
  }

  // 获取队列大小
  size_t size() {
    std::unique_lock<std::mutex> lock(mutex);
    return tasks.size();
  }
};

// 创建pipeline的函数
void create_pipeline_worker(TaskQueue& create_queue, TaskQueue& execute_queue, const hd_params& params) {
  while (true) {
    // 从创建队列获取任务
    std::shared_ptr<PipelineTask> task = create_queue.pop();
    if (!task) break; // 队列已停止

    if (params.verbosity >= 1) {
      cout << "创建Pipeline: " << task->filename << endl;
    }
    
    auto pip_create_s = std::chrono::high_resolution_clock::now();
    
    // 设置参数
    hd_params pipeline_params = params;  // 复制参数
    pipeline_params.sigproc_file = task->filename.c_str();
    
    // 创建数据源
    task->data_source = new SigprocFile(pipeline_params.sigproc_file, pipeline_params.fswap);
    if (!task->data_source || task->data_source->get_error()) {
      cerr << "ERROR: Failed to open data file: " << task->filename << endl;
      continue;
    }
    
    // 设置参数
    if (!pipeline_params.override_beam) {
      if (task->data_source->get_beam() > 0)
        pipeline_params.beam = task->data_source->get_beam() - 1;
      else
        pipeline_params.beam = 0;
    }
    
    pipeline_params.f0 = task->data_source->get_f0();
    pipeline_params.df = task->data_source->get_df();
    pipeline_params.dt = task->data_source->get_tsamp();
    pipeline_params.nchans = task->data_source->get_nchan();
    pipeline_params.utc_start = task->data_source->get_utc_start();
    pipeline_params.spectra_per_second = task->data_source->get_spectra_rate();
    
    // 获取其他必要参数
    task->stride = task->data_source->get_stride();
    task->nbits = task->data_source->get_nbit();
    task->nsamps_gulp = pipeline_params.nsamps_gulp;
    
    // 检查通道数是否是16的倍数
    if (pipeline_params.nchans % 16 != 0) {
      cerr << "ERROR: Dedisp library supports multiples of 16 channels only for file: " << task->filename << endl;
      continue;
    }
    
    // 分配缓冲区
    size_t filterbank_bytes = 2 * task->nsamps_gulp * task->stride;
    task->filterbank.resize(filterbank_bytes);
    
    // 创建pipeline
    hd_error error = hd_create_pipeline(&task->pipeline, pipeline_params);
    if (error != HD_NO_ERROR) {
      cerr << "ERROR: Pipeline creation failed for file: " << task->filename << endl;
      cerr << "       " << hd_get_error_string(error) << endl;
      continue;
    }
    
    auto pip_create_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pip_create_time = pip_create_e - pip_create_s;
    
    if (params.verbosity >= 1) {
      cout << "Pipeline创建完成: " << task->filename << ", 耗时: " << pip_create_time.count() << "秒" << endl;
    }
    
    // 将任务添加到执行队列
    execute_queue.push(task);
  }
}

// 执行pipeline的函数
void execute_pipeline_worker(TaskQueue& execute_queue, const hd_params& params) {
  while (true) {
    // 从执行队列获取任务
    std::shared_ptr<PipelineTask> task = execute_queue.pop();
    if (!task) break; // 队列已停止
    
    if (params.verbosity >= 1) {
      cout << "执行Pipeline: " << task->filename << endl;
    }
    
    auto pipeline_s = std::chrono::high_resolution_clock::now();
    
    // 执行pipeline
    size_t total_nsamps = 0;
    size_t nsamps_read = task->data_source->get_data(task->nsamps_gulp, (char*)&task->filterbank[0]);
    size_t overlap = 0;
    bool stop_requested = false;
    
    while (nsamps_read && !stop_requested) {
      if (params.verbosity >= 1) {
        cout << task->filename << ": 执行pipeline处理新数据块，样本数: " << nsamps_read << endl;
      }
      
      if (params.verbosity >= 2) {
        cout << task->filename << ": nsamp_gulp=" << task->nsamps_gulp 
             << " overlap=" << overlap << " nsamps_read=" << nsamps_read 
             << " nsamps_read+overlap=" << nsamps_read+overlap << endl;
      }
      
      hd_size nsamps_processed;
      hd_error error = hd_execute(task->pipeline, &task->filterbank[0], nsamps_read+overlap, 
                              task->nbits, total_nsamps, &nsamps_processed);
      
      if (error == HD_NO_ERROR) {
        if (params.verbosity >= 1)
          cout << task->filename << ": 处理完成 " << nsamps_processed << " 个样本." << endl;
      }
      else if (error == HD_TOO_MANY_EVENTS) {
        if (params.verbosity >= 1)
          cerr << "WARNING: " << task->filename << ": hd_execute产生过多事件，部分数据被跳过" << endl;
      }
      else {
        cerr << "ERROR: " << task->filename << ": Pipeline执行失败" << endl;
        cerr << "       " << hd_get_error_string(error) << endl;
        break;
      }
      
      if (params.verbosity >= 1)
        cout << task->filename << ": Main: nsamps_processed=" << nsamps_processed << endl;
      
      total_nsamps += nsamps_processed;
      
      // 重新定位，处理未能处理的样本
      std::copy(&task->filterbank[nsamps_processed * task->stride],
                &task->filterbank[(nsamps_read+overlap) * task->stride],
                &task->filterbank[0]);
      overlap += nsamps_read - nsamps_processed;
      nsamps_read = task->data_source->get_data(task->nsamps_gulp, (char*)&task->filterbank[overlap*task->stride]);
      
      // 文件结束时，不再执行pipeline
      if (nsamps_read < task->nsamps_gulp)
        stop_requested = true;
    }
    
    // 处理最后一部分非整倍数的样本
    if (stop_requested && nsamps_read > 0) {
      if (params.verbosity >= 1)
        cout << task->filename << ": 最后一块: nsamps_read=" << nsamps_read 
             << " nsamps_gulp=" << task->nsamps_gulp << " overlap=" << overlap << endl;
      
      hd_size nsamps_processed;
      hd_size nsamps_to_process = nsamps_read + overlap;
      if (nsamps_to_process > task->nsamps_gulp)
        nsamps_to_process = task->nsamps_gulp;
      
      hd_error error = hd_execute(task->pipeline, &task->filterbank[0], nsamps_to_process, 
                              task->nbits, total_nsamps, &nsamps_processed);
      
      if (params.verbosity >= 1)
        cout << task->filename << ": 最后一块: nsamps_processed=" << nsamps_processed << endl;
      
      if (error == HD_NO_ERROR) {
        if (params.verbosity >= 1)
          cout << task->filename << ": 处理完成 " << nsamps_processed << " 个样本." << endl;
      }
      else if (error == HD_TOO_MANY_EVENTS) {
        if (params.verbosity >= 1)
          cerr << "WARNING: " << task->filename << ": hd_execute产生过多事件，部分数据被跳过" << endl;
      }
      else if (error == HD_TOO_FEW_NSAMPS) {
        if (params.verbosity >= 1)
          cerr << "WARNING: " << task->filename << ": hd_execute没有足够的样本进行处理" << endl;
      }
      else {
        cerr << "ERROR: " << task->filename << ": Pipeline执行失败" << endl;
        cerr << "       " << hd_get_error_string(error) << endl;
      }
      
      total_nsamps += nsamps_processed;
    }
    
    auto pipeline_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pipeline_time = pipeline_e - pipeline_s;
    
    if (params.verbosity >= 1) {
      cout << task->filename << ": Pipeline执行完成，总共处理 " << total_nsamps 
           << " 个样本，耗时: " << pipeline_time.count() << "秒" << endl;
    }
  }
}

// 检查文件是否为.fil格式
bool is_filterbank_file(const std::string &path) {
  return path.size() >= 4 && path.substr(path.size() - 4) == ".fil";
}

int main(int argc, char* argv[]) 
{
 
 
 omp_set_nested(1);
  auto start = std::chrono::high_resolution_clock::now();
  auto pre_s = std::chrono::high_resolution_clock::now();
  
  // 解析命令行参数
  hd_params params;
  hd_set_default_params(&params);
  int ok = hd_parse_command_line(argc, argv, &params);
  
  if (ok < 0)
    return 1;
  
  // 获取CPU核心数，用于确定线程数
  unsigned int num_cores = std::thread::hardware_concurrency();
  unsigned int num_create_threads = std::max(1u, num_cores / 4); // 25%的核心用于创建pipeline
  unsigned int num_execute_threads = std::max(1u, num_cores - num_create_threads); // 剩余核心用于执行pipeline

  num_create_threads = 2;
  num_execute_threads = 8;
  
  if (params.verbosity >= 1) {
    cout << "检测到 " << num_cores << " 个CPU核心" << endl;
    cout << "使用 " << num_create_threads << " 个线程创建Pipeline" << endl;
    cout << "使用 " << num_execute_threads << " 个线程执行Pipeline" << endl;
  }
  
  // 创建任务队列
  TaskQueue create_queue;
  TaskQueue execute_queue;
  
  // 获取文件列表
  std::vector<std::string> files_to_process;
  
  // 如果指定了单个文件，检查是目录还是文件
  if (params.sigproc_file != nullptr) {
    std::filesystem::path path(params.sigproc_file);
    
    if (std::filesystem::is_directory(path)) {
      // 目录情况，收集所有.fil文件
      for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && is_filterbank_file(entry.path().string())) {
          files_to_process.push_back(entry.path().string());
        }
      }
      
      if (files_to_process.empty()) {
        cerr << "ERROR: 没有在目录 " << params.sigproc_file << " 中找到.fil文件" << endl;
        return 1;
      }
      
      if (1) {
        cout << "在目录 " << params.sigproc_file << " 中找到 " << files_to_process.size() << " 个.fil文件" << endl;
      }
    } else if (std::filesystem::is_regular_file(path)) {
      // 单文件情况
      files_to_process.push_back(std::string(params.sigproc_file));
    } else {
      cerr << "ERROR: 路径 " << params.sigproc_file << " 既不是文件也不是目录" << endl;
      return 1;
    }
  } else {
    cerr << "ERROR: 未指定输入文件或目录" << endl;
    hd_print_usage();
    return 1;
  }
  
  auto pre_e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> pre_time = pre_e - pre_s;
  cout << "预处理时间: " << pre_time.count() << "秒" << endl;
  
  // 创建工作线程
  std::vector<std::thread> create_threads;
  std::vector<std::thread> execute_threads;

  auto create_threads_s = std::chrono::high_resolution_clock::now();
  // 启动创建Pipeline的线程
  for (unsigned int i = 0; i < num_create_threads; i++) {
    create_threads.emplace_back(create_pipeline_worker, std::ref(create_queue), 
                               std::ref(execute_queue), std::ref(params));
  }
  
  // 启动执行Pipeline的线程
  for (unsigned int i = 0; i < num_execute_threads; i++) {
    execute_threads.emplace_back(execute_pipeline_worker, 
                                std::ref(execute_queue), std::ref(params));
  }
  
  // 添加文件到创建队列
  for (const auto& file : files_to_process) {
    auto task = std::make_shared<PipelineTask>();
    task->filename = file;
    printf("添加文件到创建队列: %s\n", file.c_str());
    create_queue.push(task);
  }
  
  // 等待所有文件加入队列后停止创建队列
  create_queue.stop();
  
  // 等待所有创建线程完成
  for (auto& thread : create_threads) {
    thread.join();
  }
  auto create_threads_e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> create_threads_time = create_threads_e - create_threads_s;
  cout << "创建Pipeline线程完成，耗时: " << create_threads_time.count() << "秒" << endl;
  
  // 等待执行队列为空后停止执行队列
  while (!execute_queue.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  execute_queue.stop();
  
  // 等待所有执行线程完成
  for (auto& thread : execute_threads) {
    thread.join();
  }
  
  if (params.verbosity >= 1) {
    cout << "所有文件处理完成" << endl;
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  cout << "总运行时间: " << elapsed.count() << "秒" << endl;
  
  return 0;
}