
#pragma once

#include <thrust/system/cuda/vector.h>

#include <map>
#include <unordered_map>
#include <stdexcept>

#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <sstream>


struct not_my_pointer : public std::exception
{
    explicit not_my_pointer(void *p)
    {
        std::stringstream s;
        s << "Pointer" << p << " was not allocated by this allocator.";
        message = s.str();
    }
    virtual ~not_my_pointer() noexcept {}
    virtual const char *what() const noexcept override
    {
        return message.c_str();
    }

private:
    std::string message;
};
struct cached_allocator
{
    typedef char value_type;
    cached_allocator() {}
    ~cached_allocator()
    {
        free_all();
    }
    char *allocate(std::ptrdiff_t num_bytes)
    {
        char *result = nullptr;
        // 从空闲块队列中寻找合适的块
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = free_blocks.find(num_bytes);
        if (it != free_blocks.end() && !it->second.empty())
        {
            result = it->second.back();
            it->second.pop_back();
            if (it->second.empty())
                free_blocks.erase(it);
        }
        else
        {
            // 没有找到合适大小的缓存块，申请新内存
            if (cudaMalloc((void **)&result, num_bytes) != cudaSuccess)
            {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
        // 记录已分配的块
        allocated_blocks[result] = num_bytes;
        return result;
    }

    void deallocate(char *ptr, size_t)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end())
        {
            throw not_my_pointer(reinterpret_cast<void *>(ptr));
        }
        std::ptrdiff_t num_bytes = it->second;
        allocated_blocks.erase(it);

        // 将释放的内存块加入空闲队列
        free_blocks[num_bytes].push_back(ptr);
    }

private:
    std::unordered_map<std::ptrdiff_t, std::vector<char *>> free_blocks; // 按大小记录空闲块
    std::unordered_map<char *, std::ptrdiff_t> allocated_blocks;         // 记录已分配块
    std::mutex mutex_;
    void free_all()
    {
        // 释放空闲块队列和已分配块队列
        for (auto &pair : free_blocks)
        {
            for (char *ptr : pair.second)
            {
                cudaFree(ptr);
            }
        }
        free_blocks.clear();

        for (auto &pair : allocated_blocks)
        {
            cudaFree(pair.first);
        }
        allocated_blocks.clear();
    }
};