/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/default_params.h"

void hd_set_default_params(hd_params* params) {
	// Default parameters
	params->verbosity       = 0;
#ifdef HAVE_PSRDADA
	params->dada_id         = 0;
#endif
	params->sigproc_file    = NULL;
	params->yield_cpu       = false;
	params->nsamps_gulp     = 262144;//131072; // TODO: Check that this is good
	// TODO: This is no longer being used
	params->dm_gulp_size    = 2048;//256;    // TODO: Check that this is good
	params->baseline_length = 2.0;
	params->beam            = 0;
	params->override_beam   = false;
	params->nchans          = 1024;
	params->dt              = 64e-6;
	params->f0              = 1581.804688;
	params->df              = -.390625;
	params->dm_min          = 0.0;
	params->dm_max          = 1000.0;
	params->dm_tol          = 1.25;
	params->dm_pulse_width  = 40;//e-6; // The expected intrinsic width of the pulse signal in microseconds
	params->dm_nbits        = 32;//8;
	params->use_scrunching  = true;
	params->scrunch_tol     = 1.15;
	params->rfi_tol         = 5.0;//1e-6;//1e-9; TODO: Should this be a probability instead?
	params->rfi_narrow      = true;
	params->rfi_broad       = true;
	params->rfi_min_beams   = 8;
	params->boxcar_max      = 4096;//2048;//512;
	params->detect_thresh   = 6.0;
	params->cand_sep_time   = 3;
	// Note: These have very little effect on the candidates, but could be important
	//         to capture (*rare*) coincident events.
	params->cand_sep_filter = 3;  // Note: filter numbers, not actual width
	params->cand_sep_dm     = 200; // Note: trials, not actual DM
	params->cand_rfi_dm_cut = 1.5;
	//params->cand_min_members = 3;
  
  // TODO: This still needs tuning!
  params->max_giant_rate  = 0;      // Max allowed giants per minute, 0 == no limit

  params->min_tscrunch_width = 4096; // Filter width at which to begin tscrunching

  params->num_channel_zaps = 0;
  params->channel_zaps = NULL;

  params->coincidencer_host = NULL;
  params->coincidencer_port = -1;

  params->fswap = false;
  params->boxcar_renorm = false;
	
	// TESTING
	//params->first_beam = 0;
	params->beam_count = 13;
	params->gpu_id = 0;
	params->utc_start = 0;
	params->spectra_per_second = 0;
	params->output_dir = ".";

	params->num_threads = 1;
}
