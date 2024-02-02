/* -*- c++ -*- */
/*
 * Copyright 2024 Harshad Sathaye.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CIRCULAR_CONV_CIRCULAR_CONV_IMPL_H
#define INCLUDED_CIRCULAR_CONV_CIRCULAR_CONV_IMPL_H

#include <gnuradio/circular_conv/circular_conv.h>
#include <gnuradio/thread/thread.h>
#include <volk/volk.h>
#include <volk_gnsssdr/volk_gnsssdr.h>
#include <numeric>
#include <armadillo>
#include <fstream>

namespace gr {
namespace circular_conv {

//using fft_complex_fwd = gr::fft::fft; // fft::fft_complex_fwd;
//using fft_complex_rev = gr::fft::fft; // fft::fft_complex_rev;
using namespace std;

class circular_conv_impl : public circular_conv
{
private:
    // Nothing to declare in this block.
    int d_code_id;
    int d_code_len;
    int d_samp_per_code;

    bool d_worker_active;
    bool d_active;

    uint16_t d_state;
    uint16_t d_dump_number;
    uint32_t d_buffer_count;
    uint32_t d_consumed_samples;
    uint32_t d_num_doppler_bins;
    uint32_t d_num_noncoherent_integrations_counter;
    uint32_t d_fft_size;
    uint32_t d_peak_algo;

    int32_t d_doppler_center;

    float d_pfa;
    float d_mag;
    float d_samp_rate;
    int d_freq_step;
    float d_chip_rate;
    int d_freq_start;
    int d_freq_stop;
    float d_input_power;
    float d_threshold;
    float d_test_statistics;

    double d_delay_samp;
    double d_doppler_hz;

    uint64_t d_acq_samplestamp;
    uint64_t d_sample_counter;

    vector<float> d_tmp_buffer;
    vector<gr_complex> d_fft_codes;
    vector<gr_complex> d_input_signal;
    vector<gr_complex> d_data_buffer;
    vector<vector<gr_complex>> d_grid_doppler_wipeoffs;
    vector<vector<float>> d_magnitude_grid;

    gr_complex *code_copy;

    gr::fft::fft_complex_fwd *d_fft_if;
    gr::fft::fft_complex_rev *d_ifft;

    void generate_sampled_code_fc(gr_complex *dest, int prn);
    void gold_code_gen(int *ca, int prn, int code_len);
    void acquisition_core(uint64_t samp_count);
    float max_to_input_power_statistic(uint32_t& indext, int32_t& doppler, uint32_t num_doppler_bins, int32_t doppler_max, int32_t doppler_step);
    void calculate_threshold();
    float first_vs_second_peak_statistic(uint32_t& indext, int32_t& doppler, uint32_t num_doppler_bins, int32_t doppler_max, int32_t doppler_step);
    void dump_results(int32_t effective_fft_size);
    void reset_acq_variables();

    ofstream wf;
    arma::fmat d_grid;

public:
    circular_conv_impl(int code_id, int peak_algo, int code_len, float samp_rate, float chip_rate, uint32_t fft_size, float freq_step, float freq_start, float freq_stop, float pfa);
    ~circular_conv_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
    
};

} // namespace circular_conv
} // namespace gr

constexpr double TWO_PI = (6.2831853071796);

#endif /* INCLUDED_CIRCULAR_CONV_CIRCULAR_CONV_IMPL_H */
