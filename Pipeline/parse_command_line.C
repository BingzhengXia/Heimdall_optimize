/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
#include <sstream>
#include <string.h>
using std::string;
#include <cstdlib>

#include "hd/parse_command_line.h"
#include "hd/default_params.h"
#include "hd/maths.h"

int hd_parse_command_line(int argc, char* argv[], hd_params* params)
{
  // TODO: Make this robust to malformed input
  size_t i=0;
  while( ++i < (size_t)argc ) {
    if( argv[i] == string("-h") ) {
      hd_print_usage();  
      return -1;
    }
    if( argv[i] == string("-v") ) {
      params->verbosity = max(params->verbosity, 1);
    }
    else if( argv[i] == string("-V") ) {
      params->verbosity = max(params->verbosity, 2);
    }
    else if( argv[i] == string("-g") ) {
      params->verbosity = max(params->verbosity, 3);
    }
    else if( argv[i] == string("-G") ) {
      params->verbosity = max(params->verbosity, 4);
    }
#ifdef HAVE_PSRDADA
    else if( argv[i] == string("-k") ) {
      sscanf(argv[++i], "%x", &(params->dada_id));
    }
#endif
    else if( argv[i] == string("-f") ) {
      params->sigproc_file = strdup(argv[++i]);
    }
    else if( argv[i] == string("-yield_cpu") ) {
      params->yield_cpu = true;
    }
    else if( argv[i] == string("-nsamps_gulp") ) {
      params->nsamps_gulp = atoi(argv[++i]);
    }
    else if( argv[i] == string("-baseline_length") ) {
      params->baseline_length = atof(argv[++i]);
    }
    else if( argv[i] == string("-dm") ) {
      params->dm_min = atof(argv[++i]);
      params->dm_max = atof(argv[++i]);
    }
    else if( argv[i] == string("-dm_tol") ) {
      params->dm_tol = atof(argv[++i]);
    }
    else if( argv[i] == string("-dm_pulse_width") ) {
      params->dm_pulse_width = atof(argv[++i]);
    }
    else if( argv[i] == string("-dm_nbits") ) {
      params->dm_nbits = atoi(argv[++i]);
    }
    else if( argv[i] == string("-gpu_id") ) {
      params->gpu_id = atoi(argv[++i]);
    }
    else if( argv[i] == string("-no_scrunching") ) {
      params->use_scrunching = false;
    }
    else if( argv[i] == string("-scrunch_tol") ) {
      params->scrunch_tol = atof(argv[++i]);
    }
    else if( argv[i] == string("-rfi_tol") ) {
      params->rfi_tol = atof(argv[++i]);
    }
    else if( argv[i] == string("-rfi_min_beams") ) {
      params->rfi_min_beams = atoi(argv[++i]);
    }
    else if( argv[i] == string("-rfi_no_narrow") ) {
      params->rfi_narrow = false;
    }
    else if( argv[i] == string("-rfi_no_broad") ) {
      params->rfi_broad = false;
    }
    else if( argv[i] == string("-boxcar_max") ) {
      params->boxcar_max = atoi(argv[++i]);
    }
    else if( argv[i] == string("-detect_thresh") ) {
      params->detect_thresh = atof(argv[++i]);
    }
    else if( argv[i] == string("-beam") ) {
      params->beam = atoi(argv[++i]) - 1;
      params->override_beam = true;
    }
    else if( argv[i] == string("-beam_count") ) {
      params->beam_count = atoi(argv[++i]);
    }
    else if( argv[i] == string("-cand_sep_time") ) {
      params->cand_sep_time = atoi(argv[++i]);
    }
    else if( argv[i] == string("-cand_sep_filter") ) {
      params->cand_sep_filter = atoi(argv[++i]);
    }
    else if( argv[i] == string("-cand_sep_dm_trial") ) {
      params->cand_sep_dm = atoi(argv[++i]);
    }
    else if( argv[i] == string("-cand_rfi_dm_cut") ) {
      params->cand_rfi_dm_cut = atof(argv[++i]);
    }
    else if( argv[i] == string("-max_giant_rate") ) {
      params->max_giant_rate = atof(argv[++i]);
    }
    else if( argv[i] == string("-output_dir") ) {
      params->output_dir = strdup(argv[++i]);
    }
    else if( argv[i] == string("-min_tscrunch_width") ) {
      params->min_tscrunch_width = atoi(argv[++i]);
    }
    else if( argv[i] == string("-coincidencer") ) {
      std::istringstream iss (argv[++i], std::istringstream::in);
      string host, port;
      getline(iss, host, ':');
      getline(iss, port); 
      params->coincidencer_host = strdup(host.c_str());
      params->coincidencer_port = atoi(port.c_str());
    }
    else if ( argv[i] == string("-fswap") ) {
      params->fswap = true;
    }
    else if ( argv[i] == string("-boxcar_renorm") ) {
      params->boxcar_renorm = true;
    }
    else if( argv[i] == string("-zap_chans") ) {
      unsigned int izap = params->num_channel_zaps;
      params->num_channel_zaps++;
      params->channel_zaps = (hd_range_t *) realloc (params->channel_zaps, sizeof(hd_range_t) * params->num_channel_zaps);
      params->channel_zaps[izap].start = atoi(argv[++i]);
      params->channel_zaps[izap].end   = atoi(argv[++i]);
    }
    else if( argv[i] == string("-t") ) {
      sscanf(argv[++i], "%d", &(params->num_threads));
    }
    else {
      cerr << "WARNING: Unknown parameter '" << argv[i] << "'" << endl;
    }
  }

  if (params->sigproc_file == NULL)
  {
#ifdef HAVE_PSRDADA
    if (params->dada_id != 0)
      return 0;
#endif
    cerr << "ERROR: no input mechanism specified" << endl;
    hd_print_usage();
    return -1;
  }
  else
    return 0;
}

void hd_print_usage()
{
  hd_params p;
  hd_set_default_params(&p);

  cout << "Usage: heimdall [options]" << endl;
  cout << "    -k  key                  use PSRDADA hexidecimal key" << endl;
  cout << "    -f  filename             process specified SIGPROC filterbank file" << endl;
  cout << "    -vVgG                    increase verbosity level" << endl;
  cout << "    -yield_cpu               yield CPU during GPU operations" << endl;
  cout << "    -gpu_id ID               run on specified GPU" << endl;
  cout << "    -nsamps_gulp num         number of samples to be read at a time [" << p.nsamps_gulp << "]" << endl;
  cout << "    -baseline_length num     number of seconds over which to smooth the baseline [" << p.baseline_length << "]" << endl;
  cout << "    -beam ##                 over-ride beam number" << endl;
  cout << "    -output_dir path         create all output files in specified path" << endl;
  cout << "    -dm min max              min and max DM" << endl;
  cout << "    -dm_tol num              SNR loss tolerance between each DM trial [" << p.dm_tol << "]" << endl;
  cout << "    -coincidencer host:port  connect to the coincidencer on the specified host and port" << endl;
  cout << "    -zap_chans start end     zap all channels between start and end channels inclusive" << endl;
  cout << "    -max_giant_rate nevents  limit the maximum number of individual detections per minute to nevents" << endl;
  cout << "    -dm_pulse_width num      expected intrinsic width of the pulse signal in microseconds" << endl;
  cout << "    -dm_nbits num            number of bits per sample in dedispersed time series [" << p.dm_nbits << "]" << endl;
  cout << "    -no_scrunching           don't use an adaptive time scrunching during dedispersion" << endl;
  cout << "    -scrunching_tol num      smear tolerance factor for time scrunching [" << p.scrunch_tol << "]" << endl;
  cout << "    -rfi_tol num             RFI exicision threshold limits [" << p.rfi_tol << "]" << endl;
  cout << "    -rfi_no_narrow           disable narrow band RFI excision" << endl;
  cout << "    -rfi_no_broad            disable 0-DM RFI excision" << endl;
  cout << "    -boxcar_max num          maximum boxcar width in samples [" << p.boxcar_max << "]" << endl;
  cout << "    -fswap                   swap channel ordering for negative DM - SIGPROC 2,4 or 8 bit only" << endl;
  cout << "    -boxcar_renorm           renormalise the boxcar filtered timeseries instead of rescale" << endl;
  cout << "    -min_tscrunch_width num  vary between high quality (large value) and high performance (low value)" << endl;
  cout << "    -t num  set number of thread and stream" << endl;
  
}
