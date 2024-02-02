/* -*- c++ -*- */
/*
 * Copyright 2024 Harshad Sathaye.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <gnuradio/fft/fft.h>
#include "circular_conv_impl.h"
#include <gnuradio/io_signature.h>
#include <boost/math/special_functions/gamma.hpp>
#include <volk_gnsssdr/volk_gnsssdr.h>
#include <matio.h>

const auto AUX_CEIL = [](float x) { return static_cast<int32_t>(static_cast<int64_t>((x) + 1)); };

namespace gr {
namespace circular_conv {

using input_type = gr_complex;
using output_type = gr_complex;

using namespace std;

circular_conv::sptr circular_conv::make(int code_id, int peak_algo, int code_len, float samp_rate, float chip_rate, uint32_t fft_size, float freq_step, float freq_start, float freq_stop, float pfa)
{
    return gnuradio::make_block_sptr<circular_conv_impl>(code_id, peak_algo, code_len, samp_rate, chip_rate, fft_size, freq_step, freq_start, freq_stop, pfa);
}


/*
 * The private constructor
 */
circular_conv_impl::circular_conv_impl(int code_id, int peak_algo, int code_len, float samp_rate, float chip_rate, uint32_t fft_size, float freq_step, float freq_start, float freq_stop, float pfa)
    : gr::sync_block("circular_conv",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
    wf = ofstream("test_statistics", ios::out | ios::binary);

    d_code_id = code_id;
    d_code_len = code_len;
    d_samp_rate = samp_rate;
    d_freq_step = static_cast<int>(freq_step);
    d_chip_rate = chip_rate;
    d_freq_start = static_cast<int>(freq_start);
    d_freq_stop = static_cast<int>(freq_stop);

    d_samp_per_code = static_cast<int32_t>(samp_rate / (chip_rate / code_len));
    d_fft_size = fft_size;
    d_peak_algo = peak_algo;

    d_logger->alert("Samp per code: {}", d_samp_per_code);

    code_copy = (gr_complex*)(d_samp_per_code * sizeof(gr_complex));

    d_tmp_buffer = vector<float>(d_fft_size);
    d_fft_codes = vector<gr_complex>(d_fft_size);
    d_input_signal = vector<gr_complex>(d_fft_size);

    d_fft_if = new fft::fft_complex_fwd(d_fft_size);
    d_ifft = new gr::fft::fft_complex_rev(d_fft_size);

    d_worker_active = false;
    d_active = true;
    d_state = 0;
    d_sample_counter = 0ULL;
    d_consumed_samples = d_fft_size;
    d_mag = 0.0;
    d_pfa = pfa;
    d_input_power = 0.0;
    d_threshold = 0.0;
    d_dump_number = 0;

    d_grid = arma::fmat();

    // Init acquisition
    d_doppler_hz = 0;
    d_delay_samp = 0;
    d_acq_samplestamp = 0ULL;
    d_num_doppler_bins = static_cast<uint32_t>(std::ceil(static_cast<double>(static_cast<int32_t>(d_freq_stop) - static_cast<int32_t>(d_freq_start)) / static_cast<double>(d_freq_step)));
    d_doppler_center = static_cast<int32_t>((d_freq_stop - d_freq_start)/2);
    d_buffer_count = 0;

    d_data_buffer = vector<gr_complex>(d_consumed_samples);
    d_magnitude_grid = vector<vector<float>>(d_num_doppler_bins, vector<float>(d_fft_size));
    d_grid_doppler_wipeoffs = vector<vector<gr_complex>>(d_num_doppler_bins, vector<gr_complex>(d_fft_size));
    
    for (uint32_t doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
        {
            std::fill(d_magnitude_grid[doppler_index].begin(), d_magnitude_grid[doppler_index].end(), 0.0);
        }

    // Update local carrier
    int32_t c_doppler;
    for (uint32_t doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
        {
            c_doppler = -static_cast<int32_t>(d_freq_stop) + 0 + d_freq_step * doppler_index;
            float phase_step_rad;
            phase_step_rad = static_cast<float>(TWO_PI) * static_cast<float>(c_doppler) / static_cast<float>(d_samp_rate);
            
            std::array<float, 1> _phase{};
            volk_gnsssdr_s32f_sincos_32fc(d_grid_doppler_wipeoffs[doppler_index].data(), -phase_step_rad, _phase.data(), d_grid_doppler_wipeoffs[doppler_index].size());

        }

    calculate_threshold();
    generate_sampled_code_fc(code_copy, d_code_id);

    d_grid = arma::fmat(d_fft_size, d_num_doppler_bins, arma::fill::zeros);
}

/*
 * Our virtual destructor.
 */
circular_conv_impl::~circular_conv_impl() {}

void circular_conv_impl::calculate_threshold()
{
    const auto effective_fft_size = static_cast<int>(d_fft_size);
    const int num_doppler_bins = d_num_doppler_bins;

    const int num_bins = effective_fft_size * num_doppler_bins;

    d_threshold = static_cast<float>(2.0 * boost::math::gamma_p_inv(2.0, std::pow(1.0 - d_pfa, 1.0 / static_cast<float>(num_bins))));
    d_logger->alert("Threshold set to - {}, Num of bins: {}", d_threshold, num_bins);
}

void circular_conv_impl::gold_code_gen(int *ca, int prn, int code_len)
{
    int delay[] = {5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252,
                   254, 255, 256, 257, 258, 469, 470, 471, 472, 473, 474,
                   509, 512, 513, 514, 515, 516, 859, 860, 861, 862};

    int g1[code_len], g2[code_len];
    int r1[10], r2[10];
    int c1, c2;
    int i, j;

    if (prn < 1 || prn > 32)
        return;

    for (i = 0; i < 10; i++)
        r1[i] = r2[i] = -1;

    for (i = 0; i < code_len; i++)
    {
        g1[i] = r1[9];
        g2[i] = r2[9];
        c1 = r1[2] * r1[9];
        c2 = r2[1] * r2[2] * r2[5] * r2[7] * r2[8] * r2[9];

        for (j = 9; j > 0; j--)
        {
            r1[j] = r1[j - 1];
            r2[j] = r2[j - 1];
        }
        r1[0] = c1;
        r2[0] = c2;
    }
    d_logger->alert("f");
    for (i = 0, j = code_len - delay[prn - 1]; i < code_len; i++, j++)
        ca[i] = (1 - g1[i] * g2[j % code_len]) / 2;

    return;
}

float circular_conv_impl::max_to_input_power_statistic(uint32_t& indext, int32_t& doppler, uint32_t num_doppler_bins, int32_t doppler_max, int32_t doppler_step)
{
    float grid_maximum = 0.0;
    uint32_t index_doppler = 0U;
    uint32_t tmp_intex_t = 0U;
    uint32_t index_time = 0U;
    const uint32_t effective_fft_size = d_fft_size;

    // Find the correlation peak and the carrier frequency
    for (uint32_t i = 0; i < num_doppler_bins; i++)
        {
            volk_gnsssdr_32f_index_max_32u(&tmp_intex_t, d_magnitude_grid[i].data(), effective_fft_size);
            if (d_magnitude_grid[i][tmp_intex_t] > grid_maximum)
                {
                    grid_maximum = d_magnitude_grid[i][tmp_intex_t];
                    index_doppler = i;
                    index_time = tmp_intex_t;
                }
        }
    indext = index_time;

    const auto index_opp = (index_doppler + d_num_doppler_bins / 2) % d_num_doppler_bins;
    d_input_power = static_cast<float>(std::accumulate(d_magnitude_grid[index_opp].data(), d_magnitude_grid[index_opp].data() + effective_fft_size, static_cast<float>(0.0)) / effective_fft_size / 2.0 / d_num_noncoherent_integrations_counter);
    doppler = -static_cast<int32_t>(doppler_max) + d_doppler_center + doppler_step * static_cast<int32_t>(index_doppler);

    return grid_maximum / d_input_power;
}

// Taken from GNSS-SDR
void circular_conv_impl::generate_sampled_code_fc(gr_complex *dest, int prn)
{
    d_logger->warn("Generating sampled code: {} chips @ {} c/s - {} s/code", d_code_len, d_chip_rate, d_samp_per_code);
    gr::thread::scoped_lock lock(d_setlock);

    int *code_int = (int *)malloc(d_code_len * sizeof(int)); 
    gr_complex *temp_fc_code = (gr_complex *)malloc(d_code_len * sizeof(gr_complex));

    float tc = 1.0F / static_cast<float>(d_chip_rate);  // C/A chip period in sec
    float ts = 1.0F / static_cast<float>(d_samp_rate);  // Sampling period in sec

    int32_t codeValueIndex;
    float aux;
    
    //gold_code_gen(code_int, prn, d_code_len);
    // int code_int[128];
    // if (prn == 1)
    //     int code_int[128] = {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0};
    // else
        
    int code1[128] = {1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0};
    int code2[128] = {1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0};

    for (int ii = 0; ii < d_code_len; ++ii)
    {   
        if (prn == 1)
            code_int[ii] = code1[ii];
        else if (prn == 2)
            code_int[ii] = code2[ii];
    }
    
    for (int ii = 0; ii < d_code_len; ++ii)
    {
        temp_fc_code[ii] = gr_complex(static_cast<float>(code_int[ii]), static_cast<float>(code_int[ii]));
        cerr << code_int[ii];
    }
    
    dest = (gr_complex*)malloc(d_samp_per_code*sizeof(gr_complex));
    
    for (int32_t i = 0; i < d_samp_per_code; i++)
    {
        // === Digitizing ==================================================

        // --- Make index array to read C/A code values --------------------
        // The length of the index array depends on the sampling frequency -
        // number of samples per millisecond (because one C/A code period is one
        // millisecond).

        aux = (ts * (static_cast<float>(i) + 1)) / tc;
        codeValueIndex = AUX_CEIL(aux) - 1;

        // --- Make the digitized version of the C/A code -------------------
        // The "upsampled" code is made by selecting values form the CA code
        // chip array (caCode) for the time instances of each sample.
        if (i == d_samp_per_code - 1)
            {
                // --- Correct the last index (due to number rounding issues)
                dest[i] = temp_fc_code[d_code_len - 1];
            }
        else
            {
                dest[i] = temp_fc_code[codeValueIndex];  // repeat the chip -> upsample
            }

    }
    
    std::copy(dest, dest + d_samp_per_code, d_fft_if->get_inbuf());
    d_fft_if->execute();  // We need the FFT of local code
    volk_32fc_conjugate_32fc(d_fft_codes.data(), d_fft_if->get_outbuf(), d_fft_size);
    //d_logger->warn("d_fft_code");
}

float circular_conv_impl::first_vs_second_peak_statistic(uint32_t& indext, int32_t& doppler, uint32_t num_doppler_bins, int32_t doppler_max, int32_t doppler_step)
{
    // Look for correlation peaks in the results
    // Find the highest peak and compare it to the second highest peak
    // The second peak is chosen not closer than 1 chip to the highest peak

    float firstPeak = 0.0;
    uint32_t index_doppler = 0U;
    uint32_t tmp_intex_t = 0U;
    uint32_t index_time = 0U;

    // Find the correlation peak and the carrier frequency
    for (uint32_t i = 0; i < num_doppler_bins; i++)
        {
            volk_gnsssdr_32f_index_max_32u(&tmp_intex_t, d_magnitude_grid[i].data(), d_fft_size);
            if (d_magnitude_grid[i][tmp_intex_t] > firstPeak)
                {
                    firstPeak = d_magnitude_grid[i][tmp_intex_t];
                    index_doppler = i;
                    index_time = tmp_intex_t;
                }
        }
    indext = index_time;

    doppler = -static_cast<int32_t>(doppler_max) + d_doppler_center + doppler_step * static_cast<int32_t>(index_doppler);

    // Find 1 chip wide code phase exclude range around the peak
    int32_t excludeRangeIndex1 = index_time - d_samp_per_code/d_code_len;
    int32_t excludeRangeIndex2 = index_time + d_samp_per_code/d_code_len;

    // Correct code phase exclude range if the range includes array boundaries
    if (excludeRangeIndex1 < 0)
        {
            excludeRangeIndex1 = d_fft_size + excludeRangeIndex1;
        }
    else if (excludeRangeIndex2 >= static_cast<int32_t>(d_fft_size))
        {
            excludeRangeIndex2 = excludeRangeIndex2 - d_fft_size;
        }

    int32_t idx = excludeRangeIndex1;
    std::copy(d_magnitude_grid[index_doppler].data(), d_magnitude_grid[index_doppler].data() + d_fft_size, d_tmp_buffer.data());
    do
        {
            d_tmp_buffer[idx] = 0.0;
            idx++;
            if (idx == static_cast<int32_t>(d_fft_size))
                {
                    idx = 0;
                }
        }
    while (idx != excludeRangeIndex2);

    // Find the second highest correlation peak in the same freq. bin ---
    volk_gnsssdr_32f_index_max_32u(&tmp_intex_t, d_tmp_buffer.data(), d_fft_size);
    const float secondPeak = d_tmp_buffer[tmp_intex_t];

    // Compute the test statistics and compare to the threshold
    return firstPeak / secondPeak;
}

// Taken from GNSS-SDR
void circular_conv_impl::acquisition_core(uint64_t samp_count)
{
    // Initialize acquisition algorithm
    int32_t doppler = 0;
    uint32_t indext = 0U;
    const uint32_t effective_fft_size = d_fft_size;
   
    std::copy(d_data_buffer.data(), d_data_buffer.data() + d_consumed_samples, d_input_signal.data());

    if (d_fft_size > d_consumed_samples)
        {
            d_logger->alert("Filling it up {} 0s", d_fft_size - d_consumed_samples);
            for (uint32_t i = d_consumed_samples; i < d_fft_size; i++)
                {
                    d_input_signal[i] = gr_complex(0.0, 0.0);
                }
        }

    const gr_complex* in = d_input_signal.data();  // Get the input samples pointer

    d_mag = 0.0;
    d_num_noncoherent_integrations_counter++;
    
    for (uint32_t doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
    {

        // Remove Doppler
        volk_32fc_x2_multiply_32fc(d_fft_if->get_inbuf(), in, d_grid_doppler_wipeoffs[doppler_index].data(), d_fft_size);

        // Perform the FFT-based convolution  (parallel time search)
        // Compute the FFT of the carrier wiped--off incoming signal
        d_fft_if->execute();

        // Multiply carrier wiped--off, Fourier transformed incoming signal with the local FFT'd code reference
        volk_32fc_x2_multiply_32fc(d_ifft->get_inbuf(), d_fft_if->get_outbuf(), d_fft_codes.data(), d_fft_size);

        // Compute the inverse FFT
        d_ifft->execute();

        // Compute squared magnitude (and accumulate in case of non-coherent integration)
        const size_t offset = 0;
        if (d_num_noncoherent_integrations_counter == 1)
            {
                volk_32fc_magnitude_squared_32f(d_magnitude_grid[doppler_index].data(), d_ifft->get_outbuf() + offset, effective_fft_size);
            }
        else
            {
                volk_32fc_magnitude_squared_32f(d_tmp_buffer.data(), d_ifft->get_outbuf() + offset, effective_fft_size);
                volk_32f_x2_add_32f(d_magnitude_grid[doppler_index].data(), d_magnitude_grid[doppler_index].data(), d_tmp_buffer.data(), effective_fft_size);
            }
        
        std::copy(d_magnitude_grid[doppler_index].data(), d_magnitude_grid[doppler_index].data() + effective_fft_size, d_grid.colptr(doppler_index));
    }

    if (d_peak_algo == 1)
        d_test_statistics = max_to_input_power_statistic(indext, doppler, d_num_doppler_bins, d_freq_stop, d_freq_step);

    else if (d_peak_algo == 2)
    {
        d_test_statistics = first_vs_second_peak_statistic(indext, doppler, d_num_doppler_bins, d_freq_stop, d_freq_step);
    }

    d_delay_samp = static_cast<double>(std::fmod(static_cast<float>(indext), d_samp_per_code));
    d_doppler_hz = static_cast<double>(doppler);
    d_acq_samplestamp = samp_count;

    if (d_test_statistics > d_threshold)
    {
        d_logger->alert("\n*** Peak found *** \n Doppler: {}\n Delay: {}\n Timestamp: {}\n Test stat: {}", d_doppler_hz, d_delay_samp, d_acq_samplestamp, d_test_statistics);
        d_state = 0;  // Positive acquisition
        dump_results(effective_fft_size);
    }
    else
    {
        //d_logger->alert("Test statistics: {}", d_test_statistics);
        d_buffer_count = 0;
        d_state = 1;
    }


    wf << d_test_statistics << "," << samp_count << "," << doppler << "\n";

    // if (d_num_noncoherent_integrations_counter == 1)
    // {
    //     d_state = 1;
    // }
    // d_num_noncoherent_integrations_counter = 0U;

    // dump_results(effective_fft_size);
    // // Record results to file if required
    // if (d_dump and d_channel == d_dump_channel)
    //     {
    //         std::copy(d_magnitude_grid[doppler_index].data(), d_magnitude_grid[doppler_index].data() + effective_fft_size, d_grid.colptr(doppler_index));
    //     }
}

void circular_conv_impl::reset_acq_variables()
{
    d_doppler_hz = 0;
    d_delay_samp = 0;
    d_acq_samplestamp = 0ULL;
    d_state = 1;
    d_buffer_count = 0U;
    d_data_buffer.clear();
    d_input_signal.clear();
}

void circular_conv_impl::dump_results(int32_t effective_fft_size)
{
    d_dump_number++;
    std::string filename = "";
    filename.append("./dump_");
    filename.append(std::to_string(d_dump_number));
    filename.append(".mat");

    mat_t* matfp = Mat_CreateVer(filename.c_str(), nullptr, MAT_FT_MAT73);
    if (matfp == nullptr)
        {
            std::cout << "Unable to create or open Acquisition dump file\n";
            // d_acq_parameters.dump = false;
        }
    else
        {
            std::array<size_t, 2> dims{static_cast<size_t>(effective_fft_size), static_cast<size_t>(d_num_doppler_bins)};
            matvar_t* matvar = Mat_VarCreate("acq_grid", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims.data(), d_grid.memptr(), 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            dims[0] = static_cast<size_t>(1);
            dims[1] = static_cast<size_t>(1);
            matvar = Mat_VarCreate("doppler_max", MAT_C_INT32, MAT_T_INT32, 1, dims.data(), &d_freq_stop, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            matvar = Mat_VarCreate("doppler_step", MAT_C_INT32, MAT_T_INT32, 1, dims.data(), &d_freq_step, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            auto aux = static_cast<float>(d_doppler_hz);
            matvar = Mat_VarCreate("acq_doppler_hz", MAT_C_SINGLE, MAT_T_SINGLE, 1, dims.data(), &aux, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            aux = static_cast<float>(d_delay_samp);
            matvar = Mat_VarCreate("acq_delay_samples", MAT_C_SINGLE, MAT_T_SINGLE, 1, dims.data(), &aux, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            matvar = Mat_VarCreate("test_statistic", MAT_C_SINGLE, MAT_T_SINGLE, 1, dims.data(), &d_test_statistics, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            matvar = Mat_VarCreate("threshold", MAT_C_SINGLE, MAT_T_SINGLE, 1, dims.data(), &d_threshold, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            matvar = Mat_VarCreate("input_power", MAT_C_SINGLE, MAT_T_SINGLE, 1, dims.data(), &d_input_power, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            matvar = Mat_VarCreate("sample_counter", MAT_C_UINT64, MAT_T_UINT64, 1, dims.data(), &d_sample_counter, 0);
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
            Mat_VarFree(matvar);

            Mat_Close(matfp);
        }
}

// General work function
int circular_conv_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    auto in1 = static_cast<const gr_complex*>(input_items[0]);
    auto out = static_cast<gr_complex*>(output_items[0]);

    //gr::thread::scoped_lock lk(d_setlock);

    //std::copy(in1, in1 + noutput_items, out + noutput_items);
    // consume_each(noutput_items);

    for (int i=0; i < noutput_items; i++)
        out[i] = in1[i];

    // if (!d_active or d_worker_active)
    // {
    //     // do not consume samples while performing a non-coherent integration
    //     bool consume_samples = ((!d_active) || (d_worker_active));
    //     if (consume_samples)
    //         {
    //             d_sample_counter += static_cast<uint64_t>(noutput_items);
    //             d_logger->alert("Consuming {}", d_sample_counter);
    //             consume_each(noutput_items);
    //         }
        
    //     return 0;
    // }
    switch(d_state)
    {
        case 0:
        {
            // Reset variables
            d_logger->debug("State 0");
            reset_acq_variables();
            d_sample_counter += static_cast<uint64_t>(noutput_items);  // sample counter
            d_state = 1;
            d_buffer_count = 0U;
            consume_each(noutput_items);
            break;
        }

        case 1:
        {
            d_logger->debug("State 1");
            uint32_t buff_increment;
            const auto* in = reinterpret_cast<const gr_complex*>(input_items[0]);  // Get the input samples pointer
            if ((noutput_items + d_buffer_count) <= d_consumed_samples)
            {
                buff_increment = noutput_items;
            }
            else
            {
                buff_increment = d_consumed_samples - d_buffer_count;
            }

            //std::copy(in, in + buff_increment, d_data_buffer.begin());
            copy(&in[0], &in[buff_increment], back_inserter(d_data_buffer));

            // for (int i=0; i<buff_increment; i++)
            //     d_data_buffer.at(i) = (in[i]);

            // If buffer will be full in next iteration
            if (d_buffer_count >= d_consumed_samples)
            {
                d_logger->debug("Setting state 2"); 
                d_state = 2;
            }
            d_buffer_count += buff_increment;
            //d_logger->info("Buffer count: {}", d_data_buffer.size());
            d_sample_counter += static_cast<uint64_t>(buff_increment);

            d_logger->alert("{}: bi {} - dbc {} - noi {} - ddb {}", d_sample_counter, buff_increment, d_buffer_count, noutput_items, d_data_buffer.size());

            //consume_each();
            break;
        }

        case 2:
        {
            d_logger->debug("State 2");
            acquisition_core(d_sample_counter);
            reset_acq_variables();
            //consume_each(0);
            d_buffer_count = 0U;
            break;
        }
    } 
    //d_logger->warn("{}", d_sample_counter);
    // Tell runtime system how many output items we produced.
    consume_each(noutput_items);
    return noutput_items;
}

} /* namespace circular_conv */
} /* namespace gr */
