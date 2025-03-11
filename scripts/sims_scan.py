import gzip
import re
import math
from scipy.stats import poisson
from scipy.stats import multinomial
import numpy as np
from scipy.optimize import minimize
import csv
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import bz2
import seaborn as sns
import matplotlib.colors as mcolors

def make_data_dict_vcf(vcf_filename, popinfo_filename):
    """
    parse a vcf file and return a dictionary containing 'calls', 'context', 
    and 'segregating' keys for each SNP. 
    - 
    - calls: dictionary where the keys are the population ids, and the values
             are tuples with two entries: number of individuals with the reference 
             allele, and number of individuals with the derived allele. 
    - context: reference allele.
    - segregating: tuple with two entries: reference allele and derived allele.

    arguments:
    - vcf_filename: name of the vcf file containing SNP data.
    - popinfo_filename: file containing population info for each sample.
    - filter: if True, only include SNPs that passed the filter criteria.
    
    add another argument that works as a flag to take only whatever category of function I want (missense or synonymous)

    """
    
    # make population map dictionary
    popmap_file = open(popinfo_filename, "r")
    popmap = {}

    for line in popmap_file:
        columns = line.strip().split("\t")
        if len(columns) >= 2:
            popmap[columns[0]] = columns[1]
    popmap_file.close()

    #open files
    vcf_file = gzip.open(vcf_filename, 'rt')
    
    data_dict = {} # initialize output dictionary
    
    poplist = []
    
    for line in vcf_file:

        if line.startswith('##'): # skip vcf metainfo
            continue
        if line.startswith('#'): # read header
            header_cols = line.split()
            
            # map sample IDs to popmap file
            for sample in header_cols[9:]:
                if sample in popmap:
                    poplist.append(popmap[sample])
                else:
                    None
 
            continue

        cols = line.split("\t")
        snp_id = '-'.join(cols[:2]) # keys for data_dict: CHR_POS
        snp_dict = {} 
        
        # filter SNPs based on 'snp_type' annotation
        info_field = cols[7]
        info_field_parts = info_field.split('|')
        if len(info_field_parts) >= 2:
            annotation = info_field_parts[1]
        else: 
            annotation = 'No annotation'

        if cols[6] != 'PASS' and cols[6] != '.':
            continue
        
        # make alleles uppercase
        ref = cols[3].upper()
        alt = cols[4].upper()
        
        if ref not in ['A', 'C', 'G', 'T'] or alt not in ['A', 'C', 'G', 'T']:
            continue

        snp_dict['segregating'] = (ref, alt)
        snp_dict['context'] = '-' + ref + '-'

        calls_dict = {}
        gtindex = cols[8].split(':').index('GT') # extract the index of the GT field

        # pair each pop from poplist with the corresponding sample genotype data from cols[9:]
        for pop, sample in zip(poplist, cols[9:]):
            if pop is None:
                continue
            
            gt = sample.split(':')[gtindex] # extract genotype info
            if pop not in calls_dict:
                calls_dict[pop] = (0, 0)
            
            # count ref and alt alleles
            refcalls, altcalls = calls_dict[pop]
            refcalls += gt[::2].count('0') # gt[::2] slices the genotype and skips the '/' or '|' character
            altcalls += gt[::2].count('1')
            calls_dict[pop] = (refcalls, altcalls)

        snp_dict['calls'] = calls_dict
        snp_dict['annotation'] = annotation
        data_dict[snp_id] = snp_dict

    vcf_file.close()
    
    return data_dict


def calculate_2d_sfs(data_dict, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type, fold=True):
    """
    calculate the two-dimensional sfs 
    for two populations from a given SNP data dictionary.

    parameters:
    - data_dict: dictionary containing SNP information. Each entry includes allele counts for populations.
    - pop1: name of population 1 in the dict
    - pop2: name of population 2 in the dict
    
    returns:
    - sfs_dict: dictionary where keys are tuples (p1_freq, p2_freq) and values are counts of SNPs with those frequencies.
    
    add a 1 to the bins where I have zero counts
    """

    
    num_genomes_p1 = pop1_size*2
    num_genomes_p2 = pop2_size*2
    sfs_dict = {}
    
    for i in range(num_genomes_p1 + 1):
        for j in range(num_genomes_p2 + 1):
            sfs_dict[(i,j)] = 0 
    
    total_sites = 0

    # loop through all snps in the data_dict
    for snp_id, snp_info in data_dict.items():
        
        chr_id, pos = snp_id.split('-')
        pos = int(pos)
        
        if start_position is not None and pos < start_position:
            continue
        if end_position is not None and pos > end_position:
            continue
        
        # filter by variant type if specified
        snp_annotation = snp_info.get('annotation')
        if variant_type is not None and snp_annotation != variant_type:
            continue
        
        # get allele counts for pop1 and pop2
        pop1_calls = snp_info['calls'].get(pop1, (0, 0))  # (ref_calls, alt_calls)
        pop2_calls = snp_info['calls'].get(pop2, (0, 0))

        pop1_calls_list = list(pop1_calls)
        pop2_calls_list = list(pop2_calls)
        
        # print(pop1_calls_list)            

        # if fold:
        #     if pos % 2 == 0:
                
        #         if pop1_calls_list[1] > self.pop1_size:      
        #             oldAlt1 = pop1_calls_list[1]      
        #             pop1_calls_list[1] = pop1_calls_list[0]
        #             pop1_calls_list[0] = oldAlt1

        #             oldAlt2 = pop1_calls_list[1]      
        #             pop2_calls_list[1] = pop2_calls_list[0]            
        #             pop2_calls_list[0] = oldAlt2      

        #     else:
        #         if pop2_calls_list[1] > self.pop2_size:
        #             oldAlt1 = pop1_calls_list[1]
        #             pop1_calls_list[1] = pop1_calls_list[0]            
        #             pop1_calls_list[0] = oldAlt1

        #             oldAlt2 = pop1_calls_list[1]
        #             pop2_calls_list[1] = pop2_calls_list[0]
        #             pop2_calls_list[0] = oldAlt2      

        # alt_count_pop1 = pop1_calls_list[1]
        # alt_count_pop2 = pop2_calls_list[1]
        
        if fold:
            if pop1_calls_list[1] + pop2_calls_list[1] > pop1_size + pop2_size: 
                   oldAlt1 = pop1_calls_list[1]      
                   pop1_calls_list[1] = pop1_calls_list[0]
                   pop1_calls_list[0] = oldAlt1
                   
                   oldAlt2 = pop2_calls_list[1]      
                   pop2_calls_list[1] = pop2_calls_list[0]
                   pop2_calls_list[0] = oldAlt2
        
        alt_count_pop1 = pop1_calls_list[1]
        alt_count_pop2 = pop2_calls_list[1]

        # skip snps where both pops are missing or have no alternate alleles
        if alt_count_pop1 == 0 and alt_count_pop2 == 0:
            continue

        # increase the corresponding bin in the sfs
        sfs_dict.setdefault((alt_count_pop1, alt_count_pop2), 0) 
        sfs_dict[(alt_count_pop1, alt_count_pop2)] += 1 
        
        # increase total number of sites
        total_sites += 1
        
    # add pseudo-counts to all bins (1/total_sites)
    # pseudo_count = 0
    # if total_sites > 0:
    #     pseudo_count = 1 / total_sites
    # else:
    #     0
    
    # for key in sfs_dict.keys():
    #     sfs_dict[key] += pseudo_count

    return sfs_dict   

def normalize_2d_sfs(sfs):
    
    sfs = sfs
    
    # sum all the sites
    counts = list(sfs.values())
    total = sum(counts[1:-1]) # exclude first and last bin 
    # print(total)
    
    # divide each bin value by the total number of sites
    normalized_sfs = {}
    for coords, values in sfs.items():
        normalized_sfs[coords] = values / total
    return normalized_sfs

def count_snps(window_data, variant_type):
    
    snp_count = 0
    for snp_data in window_data.values():
        if variant_type is None:
            snp_count += 1
        elif snp_data.get("annotation") == variant_type:
            snp_count += 1
    return snp_count


def calculate_1d_sfs(data_dict, pop, pop_size, start_position, end_position, variant_type):

    num_genomes = pop_size*2
    sfs_dict = {}
    
    for i in range(num_genomes + 1):
        sfs_dict[i] = 0

    total_sites = 0
    
    for snp_id, snp_info in data_dict.items():
        chr_id, pos = snp_id.split('-')
        pos = int(pos)

        if start_position is not None and pos < start_position:
            continue
        if end_position is not None and pos > end_position:
            continue

        snp_annotation = snp_info.get('annotation')
        if variant_type is not None and snp_annotation != variant_type:
            continue

        pop_calls = snp_info['calls'].get(pop, (0, 0))  # (ref_calls, alt_calls)
        alt_count = pop_calls[1]

        if alt_count == 0:
            continue

        sfs_dict[alt_count] += 1

        total_sites += 1

    # pseudo_count = 0
    # if total_sites > 0:
    #     pseudo_count = 1 / total_sites

    # for key in sfs_dict.keys():
    #     sfs_dict[key] += pseudo_count

    return sfs_dict 


def fold_1d_sfs(sfs_dict):

    # get total number of chromosomes (2N)
    num_chromosomes = max(sfs_dict.keys())

    folded_sfs_dict = {}

    for freq, count in sfs_dict.items():
        # calculate maf
        minor_freq = min(freq, num_chromosomes - freq)

        # add the count to the corresponding bin in the folded sfs
        if minor_freq in folded_sfs_dict:
            folded_sfs_dict[minor_freq] += count
        else:
            folded_sfs_dict[minor_freq] = count

    return folded_sfs_dict


def calculate_likelihood_1D(foreground_sfs, background_sfs): 
    
    bins = sorted(foreground_sfs.keys())
    
    ''' foreground '''
    # get observed counts from foreground
    counts_fg = []
    for k in bins[1:-1]:
        count = foreground_sfs[k]
        counts_fg.append(int(count))
    # print(counts_fg)
    
    # Calculate total foreground counts
    total_fg = sum(counts_fg)

    # # Skip if total_fg is zero
    # if total_fg == 0:
    #     # print("Warning: Foreground SFS has zero counts. Skipping calculation.")
    #     return None
    
    # normalize foreground counts
    probabilities_fg_norm = []
    for count in counts_fg:
        p_norm = count/total_fg
        probabilities_fg_norm.append(p_norm)
    # print(probabilities_fg)
    
    ''' background '''
    # get observed counts from background
    counts_bg = []
    for k in bins[1:-1]:
        count_bg = background_sfs[k]
        counts_bg.append(count_bg)
    # print(probabilities_bg)
    
    # # Calculate total background counts
    # total_bg = sum(counts_bg)
    
    # # Skip if total_bg is zero
    # if total_bg == 0:
    #     # print("Warning: Background SFS has zero counts. Skipping calculation.")
    #     return None
    
    total_bg = sum(counts_bg)
    probabilities_bg_norm = []
    for count in counts_bg:
        p_norm = count/total_bg
        probabilities_bg_norm.append(p_norm)
    # print(probabilities_bg)
    
    # # probs from normalized foreground
    # foreground_sfs_norm = self.normalize_1d_sfs(foreground_sfs)
    
    # probabilities_fg = []
    # for k in bins[1:-1]:
    #     probability_fg = foreground_sfs_norm[k]
    #     probabilities_fg.append(probability_fg)
    # # print(probabilities_fg)
    
    # observed_fg = []
    # for k in bins[1:-1]:
    #     probability_fg = foreground_sfs[k]
    #     observed_fg.append(probability_fg)

    
    log_likelihood_bg = multinomial.logpmf(x=counts_fg, n=total_fg, p=probabilities_bg_norm)
    log_likelihood_fg = multinomial.logpmf(x=counts_fg, n=total_fg, p=probabilities_fg_norm)
    
    clr = 2*(log_likelihood_fg - log_likelihood_bg)
    
    return clr 


def calculate_likelihood_2D(foreground_2d_sfs, background_2d_sfs):
    
    bins = sorted(foreground_2d_sfs.keys())

    
    ''' foreground '''
    # get observed counts from foreground
    counts_fg = []
    for k in bins[1:-1]:
        count = foreground_2d_sfs[k]
        counts_fg.append(int(count))
    # print(counts_fg)
    
    # normalize foreground counts
    total_fg = sum(counts_fg)
    probabilities_fg_norm = []
    for count in counts_fg:
        p_norm = count/total_fg
        probabilities_fg_norm.append(p_norm)
    # print(probabilities_fg)
    
    ''' background '''
    # get observed counts from background
    counts_bg = []
    for k in bins[1:-1]:
        count_bg = background_2d_sfs[k]
        counts_bg.append(count_bg)
    # print(probabilities_bg)
    
    # normalize background counts
    total_bg = sum(counts_bg)
    probabilities_bg_norm = []
    for count in counts_bg:
        p_norm = count/total_bg
        probabilities_bg_norm.append(p_norm)
    # print(probabilities_bg)
    
    log_likelihood_bg = multinomial.logpmf(x=counts_fg, n=total_fg, p=probabilities_bg_norm)
    log_likelihood_fg = multinomial.logpmf(x=counts_fg, n=total_fg, p=probabilities_fg_norm)
    
    clr = 2*(log_likelihood_fg - log_likelihood_bg)
    
    return clr   

def get_gens(main_dir):
    search_strings = set()
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            parts = file.split('.')
            if len(parts) == 5:
                search_strings.add(parts[1])
    return search_strings    

def process_window(data_dict, bg_2d_sfs, bg_p1_sfs, bg_p2_sfs, window_size, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type):
    
    '''
    scan the genome in windows and calculate the p values using 'calculate_p'
    
    arguments: 
    - data_dict:
    - sfs_normalized:
    - window_size:
    - pop1:
    - pop2:
    '''
    
    # sort snps by chromosome and position
    sorted_snps = []
    for snp_key in data_dict.keys():
        coords = snp_key.split('-')
        chromosome_id = coords[0]
        position = int(coords[1])
        sorted_snps.append((chromosome_id, position, snp_key))
        
    sorted_snps.sort(key=lambda x: (x[0], x[1]))
    
    results = {}
    current_chromosome = None
    current_window_start = 0
    window_data = {}
    
    # loop over the sorted snps in windows of window_size
    for chrom, pos, snp_key in sorted_snps:
        # reset the window start for each new chromosome
        if chrom != current_chromosome:
            if window_data:
                # compute likelihood stats
                fg_2d_sfs = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                T2D = calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                 
                fg_p1_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop1, pop1_size, start_position, end_position, variant_type))
                T1D_p1 = calculate_likelihood_1D(fg_p1_sfs, bg_p1_sfs)

                fg_p2_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop2, pop2_size, start_position, end_position, variant_type))
                T1D_p2 = calculate_likelihood_1D(fg_p2_sfs, bg_p2_sfs)

                new_term_p1 = T2D - T1D_p1
                new_term_p2 = T2D - T1D_p2
                
                T2D_diff = T2D - (T1D_p1 - T1D_p2)/2
                
                snp_count = count_snps(window_data, variant_type)
                window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                results[window_range] = {
                    "window_type": "background" if 0 <= current_window_start < 500000 else "foreground",
                    "window_start": current_window_start,
                    "window_end": current_window_start + window_size,
                    "snp_count": snp_count,
                    "T2D": T2D,
                    "T1D_p1": T1D_p1,
                    "T1D_p2": T1D_p2,
                    "new_term_p1": new_term_p1,
                    "new_term_p2": new_term_p2,
                    "T2D_diff": T2D_diff
                }

            # start new chromosome
            current_chromosome = chrom
            current_window_start = 1
            window_data = {}
        
        # check if SNP is within the current window
        if pos < current_window_start + window_size:
            window_data[snp_key] = data_dict[snp_key]  # add SNP to current window
        else:
            # calculate p-values for the current window
            if window_data:
                # compute likelihood stats
                fg_2d_sfs = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                T2D = calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                 
                fg_p1_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop1, pop1_size, start_position, end_position, variant_type))
                T1D_p1 = calculate_likelihood_1D(fg_p1_sfs, bg_p1_sfs)

                fg_p2_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop2, pop2_size, start_position, end_position, variant_type))
                T1D_p2 = calculate_likelihood_1D(fg_p2_sfs, bg_p2_sfs)

                new_term_p1 = T2D - T1D_p1
                new_term_p2 = T2D - T1D_p2
                
                T2D_diff = T2D - (T1D_p1 - T1D_p2)/2
                
                snp_count = count_snps(window_data, variant_type)
                window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                results[window_range] = {
                    "window_type": "background" if 0 <= current_window_start < 500000 else "foreground",
                    "window_start": current_window_start,
                    "window_end": current_window_start + window_size,
                    "snp_count": snp_count,
                    "T2D": T2D,
                    "T1D_p1": T1D_p1,
                    "T1D_p2": T1D_p2,
                    "new_term_p1": new_term_p1,
                    "new_term_p2": new_term_p2,
                    "T2D_diff": T2D_diff
                }
            # move to the next window, aligned to the window size
            current_window_start += window_size * ((pos - current_window_start) // window_size)
            window_data = {snp_key: data_dict[snp_key]}

    # calculate for the last window if it has any data
    if window_data:
        # compute likelihood stats
        fg_2d_sfs = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
        T2D = calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
         
        fg_p1_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop1, pop1_size, start_position, end_position, variant_type))
        T1D_p1 = calculate_likelihood_1D(fg_p1_sfs, bg_p1_sfs)

        fg_p2_sfs = fold_1d_sfs(calculate_1d_sfs(window_data, pop2, pop2_size, start_position, end_position, variant_type))
        T1D_p2 = calculate_likelihood_1D(fg_p2_sfs, bg_p2_sfs)

        new_term_p1 = T2D - T1D_p1
        new_term_p2 = T2D - T1D_p2
        
        T2D_diff = T2D - (T1D_p1 - T1D_p2)/2
        
        snp_count = count_snps(window_data, variant_type)
        window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
        results[window_range] = {
            "window_type": "background" if 0 <= current_window_start < 500000 else "foreground",
            "window_start": current_window_start,
            "window_end": current_window_start + window_size,
            "snp_count": snp_count,
            "T2D": T2D,
            "T1D_p1": T1D_p1,
            "T1D_p2": T1D_p2,
            "new_term_p1": new_term_p1,
            "new_term_p2": new_term_p2,
            "T2D_diff": T2D_diff
        }
        
    return results


def likelihood_scan(main_dir, output):
    # Define pop map file
    popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

    # Get generation IDs
    generations = get_gens(main_dir)

    # Define column names
    col_names = ['generation', 'iteration', 'region', 'window_coords', 'snp_count', 'T2D', 'T1D_p1', 'T1D_p2', 'new_term_p1', 'new_term_p2', 'T2D_diff']

    with open(output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col_names)
        writer.writeheader()

        # Iterate over each generation
        for generation in generations:
            target_vcfs = glob.glob(f"{main_dir}/iter*/*{generation}*.vcf.gz")
            concatenated_vcfs = glob.glob(f"{main_dir}/concatenated_vcfs/gen.{generation}.concatenated.vcf.gz")
                
            # Get background SFS from concatenated VCFs
            for vcf in concatenated_vcfs:
                data_dict = make_data_dict_vcf(vcf, popinfo_filename)
                bg_2d_sfs = calculate_2d_sfs(data_dict, 'p1', 'p2', 5, 5, start_position=0, end_position=500000, variant_type=None)
                bg_p1_sfs = calculate_1d_sfs(data_dict, 'p1', 5, start_position=0, end_position=500000, variant_type=None)
                bg_p2_sfs = calculate_1d_sfs(data_dict, 'p2', 5, start_position=0, end_position=500000, variant_type=None)

                # Process target VCFs
                for vcf_input in target_vcfs:
                    data_dict_target = make_data_dict_vcf(vcf_input, popinfo_filename)
                    results = process_window(data_dict_target, bg_2d_sfs, bg_p1_sfs, bg_p2_sfs, 500000, 'p1', 'p2', 5, 5, start_position=None, end_position=None, variant_type=None)

                    # Extract iteration number from filename
                    iteration_number = int(vcf_input.split('.')[2])
                
                    for window_coords, result in results.items():
                        # Determine region
                        window_start, window_end = window_coords.split(' ')[1].split('-')
                        region = 'background' if int(window_end) <= 1000000 else 'foreground'

                        writer.writerow({
                            'generation': generation,
                            'iteration': iteration_number,
                            'region': region,
                            'window_coords': window_coords,
                            'snp_count': result["snp_count"],
                            'T2D': result["T2D"],
                            'T1D_p1': result["T1D_p1"],
                            'T1D_p2': result["T1D_p2"],
                            'new_term_p1': result["new_term_p1"],
                            'new_term_p2': result["new_term_p2"],
                            'T2D_diff': result["T2D_diff"]
                        })

def likelihood_scan(main_dir):
    # Define pop map file
    popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

    # Get generation IDs
    generations = get_gens(main_dir)

    # Initialize a dictionary to store results
    likelihood_results = {}

    # Get list of target vcfs and concatenated vcf files
    for generation in generations:
        target_vcfs = glob.glob(f"{main_dir}/iter*/*{generation}*.vcf.gz")
        concatenated_vcfs = glob.glob(f"{main_dir}/concatenated_vcfs/gen.{generation}.concatenated.vcf.gz")

        # Get background SFS from concatenated VCFs
        for vcf in concatenated_vcfs:
            data_dict = make_data_dict_vcf(vcf, popinfo_filename)
            bg_2d_sfs = calculate_2d_sfs(data_dict, 'p1', 'p2', 5, 5, start_position=0, end_position=500000, variant_type=None)
            bg_p1_sfs = calculate_1d_sfs(data_dict, 'p1', 5, start_position=0, end_position=500000, variant_type=None)
            bg_p2_sfs = calculate_1d_sfs(data_dict, 'p2', 5, start_position=0, end_position=500000, variant_type=None)

            # Get likelihood values using target VCFs
            for vcf_input in target_vcfs:
                data_dict_target = make_data_dict_vcf(vcf_input, popinfo_filename)
                results = process_window(data_dict_target, bg_2d_sfs, bg_p1_sfs, bg_p2_sfs, 500000, 'p1', 'p2', 5, 5, start_position=None, end_position=None, variant_type=None)

                # Extract iteration number from filename
                iteration_number = int(vcf_input.split('.')[2])

                for key, value in results.items():
                    # Determine region based on window coordinates
                    window_start, window_end = map(int, key.split(' ')[1].split('-'))
                    region = 'background' if window_end <= 1000000 else 'foreground'

                    # Store results in the dictionary
                    likelihood_results[(generation, iteration_number, key)] = {
                        'generation': generation,
                        'iteration': iteration_number,
                        'region': region,
                        'window_coords': key,
                        'likelihood': value
                    }

    return likelihood_results

sims_dir = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams"
likelihood_scan(sims_dir, "sims_dadiparams_results.csv")

sweep_sims_dir = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/sweep_results/vcfs_lowMigRate"
likelihood_scan(sweep_sims_dir, "sims_sweepLowMigRate_results.csv")



def concatenate_fst_files(path):
    
    fst_files = glob.glob(f"{path}/*/*.txt")
    # print(fst_files)
    
    first_txt = fst_files[0]
    
    with open(first_txt, 'r') as f:
        header_lines = []
        for line in f:
            if line.startswith('cycle'):
                header_lines.append(line)
                
    output_file = f"{path}/concatenated_fst.txt"
    with open(output_file, 'w') as out:
        out.writelines(header_lines)
    
        for file in fst_files:
            with open(file, 'r') as f:
                for line in f:
                    if not line.startswith('cycle'):
                        out.write(line)

concatenate_fst_files("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams")
concatenate_fst_files(sweep_sims_dir)

# normalize sfs calculated using dadi

def normalize_dadi_sfs(sfs, norm_sfs):
           
    with open(sfs, 'r') as f:
        lines = f.readlines()        
    
    allele_counts = []
    for x in lines[1].strip().split():
        allele_counts.append(float(x))
    total_count = sum(allele_counts[1:-1])
    
    normalized_counts = []
    # print(allele_counts)
    for count in allele_counts:
        normalized_count = count / total_count
        if normalized_count != 0.0:
            log_norm_count = math.log(normalized_count*1e3)
            normalized_counts.append(log_norm_count)
        else:
            normalized_counts.append(0)  # Or another appropriate value for zero counts

    print(normalized_counts)
        
    with open(norm_sfs, 'w') as f:
        f.write(lines[0])
        
        normalized_counts_str = ""
        for count in normalized_counts:
            # int_count = int(count)
            normalized_counts_str += str(count) + " "
        normalized_counts_str = normalized_counts_str.strip()
        f.write(normalized_counts_str + '\n')
        
        f.write(lines[2])

# don't include the counts at zero and the last bin
# get the -log of each bin to get the fractional numbers and it would plot 

# FST outliers 
FST_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/FST_outliers.10.folded.fs'
FST_norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/FST_outliers.10.normalized.folded.fs'
normalize_dadi_sfs(FST_sfs, FST_norm_sfs)

# T2D outliers
T2D_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/T2D_outliers.10.folded.fs'
T2D_norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/T2D_outliers.10.normalized.folded.fs'
normalize_dadi_sfs(T2D_sfs, T2D_norm_sfs)

# generation 8000 background
gen8000_bg_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/gen8000_background.10.folded.fs'
gen8000_bg_norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/gen8000_background.10.normalized.folded.fs'
normalize_dadi_sfs(gen8000_bg_sfs, gen8000_bg_norm_sfs)


# backgrounds from outlier VCFs
FST_bg_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/FST_outliers_background.10.folded.fs'
FST_bg_norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/FST_outliers_background.10.normalized.folded.fs'
normalize_dadi_sfs(FST_bg_sfs, FST_bg_norm_sfs)

T2D_bg_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/T2D_outliers_background.10.folded.fs'
T2D_bg_norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/sfs_plots/T2D_outliers_background.10.normalized.folded.fs'
normalize_dadi_sfs(T2D_bg_sfs, T2D_bg_norm_sfs)




# Plotting SFS
FST_outliers_vcf = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/merged_gen8000_FST_outliers_foregrounds.vcf.gz'
T2D_outliers_vcf = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs_dadiparams/merged_gen8000_T2D_outliers_foregrounds.vcf.gz'
popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

FST_outliers_dict = make_data_dict_vcf(FST_outliers_vcf, popinfo_filename)
T2D_outliers_dict = make_data_dict_vcf(T2D_outliers_vcf, popinfo_filename)

FST_outliers_2DSFS = calculate_2d_sfs(FST_outliers_dict, 'p1', 'p2', 5, 5, start_position=None, end_position=None, variant_type=None)
T2D_outliers_2DSFS = calculate_2d_sfs(T2D_outliers_dict, 'p1', 'p2', 5, 5, start_position=None, end_position=None, variant_type=None)

FST_outliers_2DSFS_norm = normalize_2d_sfs(FST_outliers_2DSFS)
T2D_outliers_2DSFS_norm = normalize_2d_sfs(T2D_outliers_2DSFS)

def plot_2d_sfs(sfs_dict, sample_size, vmin=None, vmax=None, ax=None,
                 pop_ids=('Pop1', 'Pop2'), colorbar=True, cmap='viridis_r', show=True):
    """
    Plots a 2D Site Frequency Spectrum (SFS) from a dictionary.
    
    Parameters:
    - sfs_dict: Dictionary with keys as (freq_pop1, freq_pop2) and values as allele counts.
    - sample_size: Tuple (n1, n2) specifying the maximum frequency range for plotting.
    - vmin, vmax: Minimum and maximum values for color scaling.
    - ax: Matplotlib Axes object. If None, a new figure is created.
    - pop_ids: Labels for the populations.
    - colorbar: Whether to display a colorbar.
    - cmap: Colormap to use for plotting.
    - show: Whether to display the plot immediately.
    
    Returns:
    - Matplotlib colorbar object if colorbar is True.
    """
    n1, n2 = sample_size
    sfs_matrix = np.zeros((n1 + 1, n2 + 1))
    
    # Populate the SFS matrix
    for (f1, f2), count in sfs_dict.items():
        if f1 <= n1 and f2 <= n2:
            sfs_matrix[f1, f2] = count
    
    if vmin is None:
        vmin = np.min(sfs_matrix[sfs_matrix > 0]) if np.any(sfs_matrix > 0) else 1
    if vmax is None:
        vmax = np.max(sfs_matrix)
    
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if vmax / vmin > 10 else mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    cax = ax.imshow(sfs_matrix.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax.set_xlabel(pop_ids[0])
    ax.set_ylabel(pop_ids[1])
    
    if colorbar:
        cb = plt.colorbar(cax, ax=ax)
        return cb
    
    if show:
        plt.show()
    
    return None

plot_2d_sfs(FST_outliers_2DSFS_norm, sample_size=(10,10))
plot_2d_sfs(T2D_outliers_2DSFS_norm, sample_size=(10,10))

plot_2d_sfs(FST_outliers_2DSFS, sample_size=(10,10))
plot_2d_sfs(T2D_outliers_2DSFS, sample_size=(10,10))




# find residuals
def calculate_residuals(sfs1, sfs2):
    residuals = {}
    
    # get all frequency bins from both dictionaries
    all_keys = set(sfs1.keys()).union(set(sfs2.keys()))
    
    for key in all_keys:
        count1 = sfs1.get(key, 0)  # Get count from SFS1, default to 0 if missing
        count2 = sfs2.get(key, 0)  # Get count from SFS2, default to 0 if missing
        
        residuals[key] = count1 - count2
        
        # # compute variance (avoid division by zero)
        # variance = (count1 + count2) / 2
        # if variance > 0:
        #     residuals[key] = (count1 - count2) / np.sqrt(variance)
        # else:
        #     residuals[key] = 0  # if variance is 0, residual is set to 0
    
    return residuals

residual_T2D_FST = calculate_residuals(T2D_outliers_2DSFS_norm, FST_outliers_2DSFS_norm)

plot_2d_sfs(residual_T2D_FST, (10,10))


def sfs_file_to_dict(file_path):
    """
    Reads a 2D SFS from a dadi-style .fs file (without using dadi) and converts it into a dictionary.

    Parameters:
        file_path (str): Path to the dadi-style SFS file.

    Returns:
        dict: A dictionary where keys are tuples (freq_pop1, freq_pop2) and values are counts.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove comments and blank lines
    data_lines = [line.strip() for line in lines if not line.startswith("//") and line.strip()]

    # Extract SFS dimensions
    header_parts = data_lines[0].split()
    try:
        dim1, dim2 = int(header_parts[0]), int(header_parts[1])  # Extract dimensions
    except ValueError:
        raise ValueError(f"Error: Could not parse SFS dimensions from header: {data_lines[0]}")

    # Read numerical values from the SFS file
    raw_values = [float(val) for line in data_lines[1:] for val in line.split()]

    # Expecting dim1 × dim2 values for the SFS matrix
    total_expected = dim1 * dim2
    if len(raw_values) < total_expected:
        raise ValueError(f"Error: Expected {total_expected} values, but found {len(raw_values)}. File may be incorrectly formatted.")

    # First `dim1 * dim2` values are the SFS matrix, remaining values (if any) are masks
    sfs_values = np.array(raw_values[:total_expected]).reshape(dim1, dim2)

    # Convert to dictionary, keeping only nonzero counts
    sfs_dict = {(i, j): sfs_values[i, j] for i in range(dim1) for j in range(dim2) if sfs_values[i, j] > 0}

    return sfs_dict


FST_outliers_sfs = sfs_file_to_dict(FST_norm_sfs)
plot_2d_sfs(FST_outliers_sfs, (10,10), vmin=0.01, vmax=7)

T2D_outliers_sfs = sfs_file_to_dict(T2D_norm_sfs)
plot_2d_sfs(T2D_outliers_sfs, (10,10), vmin=0.01, vmax=7)

