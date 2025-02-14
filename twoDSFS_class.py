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


 
class LikelihoodInference_jointSFS:
    def __init__(self, vcf_filename, popinfo_filename, start_position=None, end_position=None, 
                 pop1='uv', pop2='bv', pop1_size=18, pop2_size=14, variant_type=None, fold=True):
        self.vcf_filename = vcf_filename 
        self.popinfo_filename = popinfo_filename
        self.pop1 = pop1
        self.pop2 = pop2
        self.pop1_size = pop1_size
        self.pop2_size = pop2_size
        self.start_position = start_position
        self.end_position = end_position
        self.variant_type = variant_type
        self.fold = fold
        
    
    def make_data_dict_vcf(self):
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
        popmap_file = open(self.popinfo_filename, "r")
        popmap = {}

        for line in popmap_file:
            columns = line.strip().split("\t")
            if len(columns) >= 2:
                popmap[columns[0]] = columns[1]
        popmap_file.close()

        #open files
        vcf_file = gzip.open(self.vcf_filename, 'rt')
        
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

    def calculate_2d_sfs(self, data_dict):
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
        self.data_dict = data_dict
        
        num_genomes_p1 = self.pop1_size*2
        num_genomes_p2 = self.pop2_size*2
        sfs_dict = {}
        
        for i in range(num_genomes_p1 + 1):
            for j in range(num_genomes_p2 + 1):
                sfs_dict[(i,j)] = 0 
        
        total_sites = 0

        # loop through all snps in the data_dict
        for snp_id, snp_info in self.data_dict.items():
            
            chr_id, pos = snp_id.split('-')
            pos = int(pos)
            
            if self.start_position is not None and pos < self.start_position:
                continue
            if self.end_position is not None and pos > self.end_position:
                continue
            
            # filter by variant type if specified
            snp_annotation = snp_info.get('annotation')
            if self.variant_type is not None and snp_annotation != self.variant_type:
                continue
            
            # get allele counts for pop1 and pop2
            pop1_calls = snp_info['calls'].get(self.pop1, (0, 0))  # (ref_calls, alt_calls)
            pop2_calls = snp_info['calls'].get(self.pop2, (0, 0))

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
            
            if self.fold:
                if pop1_calls_list[1] + pop2_calls_list[1] > self.pop1_size + self.pop2_size: 
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
        pseudo_count = 0
        if total_sites > 0:
            pseudo_count = 1 / total_sites
        else:
            0
        
        for key in sfs_dict.keys():
            sfs_dict[key] += pseudo_count

        return sfs_dict    
    
    def normalize_2d_sfs(self, sfs):
        
        self.sfs = sfs
        
        # sum all the sites
        counts = list(self.sfs.values())
        total = sum(counts[1:-1]) # exclude first and last bin 
        # print(total)
        
        # divide each bin value by the total number of sites
        normalized_sfs = {}
        for coords, values in self.sfs.items():
            normalized_sfs[coords] = values / total
        return normalized_sfs
    
    def calculate_p(self, foreground_sfs, background_sfs):
        '''
        calculate the probability mass function (p values) of the foreground SFS 
        given the background SFS using poisson distribution.
        
        arguments:
        - foreground_sfs: dictionary containing the sfs of a genomic region
        - background_sfs: dictionary containing the normalized sfs of a genomic region
        '''
        
        self.foreground_sfs = foreground_sfs
        self.background_sfs = background_sfs
        
        # calculate parameters for poisson.pmf (S_w and M_dict)
        S_w = sum(foreground_sfs.values()) # total number of sites in the window across all bins
        # multiply S_w by the normalized SFS to get the expected number of sites (M)
        M_dict = {}
        for k in foreground_sfs.keys():
            normalized_value = background_sfs.get(k,0) # get the normalized SFS (background)
            M_value = S_w * normalized_value
            M_dict[k] = M_value

        # calculate the p values
        p_values = {}
        p_values_sum = 0
        
        for k in foreground_sfs.keys():
            observed_count = foreground_sfs[k]
            expected_count = M_dict.get(k,0)
            # print(observed_count)
            # print(expected_count)
            
            #skip if expected_count is zero; logpmf return infinity values that can't be added
            if expected_count == 0:
                continue
            
            p_value = poisson.logpmf(k=int(observed_count), mu=expected_count)
            p_values[k] = p_value
            p_values_sum += p_value
                
        return p_values_sum
    
    def count_snps(self, window_data, variant_type):
        
        self.window_data = window_data
        self.variant_type = variant_type
        
        snp_count = 0
        for snp_data in window_data.values():
            if variant_type is None:
                snp_count += 1
            elif snp_data.get("annotation") == variant_type:
                snp_count += 1
        return snp_count
    
    def calculate_p_window(self, data_dict, sfs_normalized, window_size, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type):
        
        '''
        scan the genome in windows and calculate the p values using 'calculate_p'
        
        arguments: 
        - data_dict:
        - sfs_normalized:
        - window_size:
        - pop1:
        - pop2:
        '''
        
        self.data_dict = data_dict
        self.sfs_normalized = sfs_normalized
        self.window_size = window_size
        self.pop1 = pop1
        self.pop2 = pop2
        self.pop1_size = pop1_size
        self.pop2_size = pop2_size
        self.start_position = start_position
        self.end_position = end_position
        self.variant_type = variant_type
        
        # sort snps by chromosome and position
        sorted_snps = []
        for snp_key in data_dict.keys():
            coords = snp_key.split('-')
            chromosome_id = coords[0]
            position = int(coords[1])
            sorted_snps.append((chromosome_id, position, snp_key))
            
        sorted_snps.sort(key=lambda x: (x[0], x[1]))
        
        window_p_values = {}
        current_chromosome = None
        current_window_start = 0
        window_data = {}
        
        # loop over the sorted snps in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            # reset the window start for each new chromosome
            if chrom != current_chromosome:
                if window_data:
                    # calculate p-values for the last window of the previous chromosome
                    sfs_dict = self.calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                    p_values_dict = self.calculate_p(sfs_dict, sfs_normalized)
                    snp_count = self.count_snps(window_data, variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    window_p_values[window_range] = {
                        "p_values": p_values_dict,
                        "snp_count": snp_count
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
                    sfs_dict = self.calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                    p_values_dict = self.calculate_p(sfs_dict, sfs_normalized)
                    snp_count = self.count_snps(window_data, variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    window_p_values[window_range] = {
                        "p_values": p_values_dict,
                        "snp_count": snp_count
                    }

                # move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}

        # calculate for the last window if it has any data
        if window_data:
            sfs_dict = self.calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
            p_values_dict = self.calculate_p(sfs_dict, sfs_normalized)
            snp_count = self.count_snps(window_data, variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            window_p_values[window_range] = {
                        "p_values": p_values_dict,
                        "snp_count": snp_count
                    }
            
        return window_p_values
    
    
    # Incorporating 1-D
    
    def calculate_1d_sfs(self, data_dict, pop, pop_size, start_position, end_position, variant_type):
        self.data_dict = data_dict
        self.pop = pop
        self.pop_size = pop_size
        self.start_position = start_position
        self.end_position = end_position
        self.variant_type = variant_type
    
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
    
        pseudo_count = 0
        if total_sites > 0:
            pseudo_count = 1 / total_sites
    
        for key in sfs_dict.keys():
            sfs_dict[key] += pseudo_count
    
        return sfs_dict        
    
    def fold_1d_sfs(self, sfs_dict):

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
    
    def normalize_1d_sfs(self, sfs):
        self.sfs = sfs
        
        counts = list(sfs.values())
        total = sum(counts[1:-1]) # exclude first and last bin 
        # print(total)
        normalized_sfs = {}
        
        for freq, values in sfs.items():
            normalized_sfs[freq] = values / total
            
        return normalized_sfs
    
    def calculate_likelihood_1D(self, foreground_sfs, background_sfs): 
        
        self.foreground_sfs = foreground_sfs
        self.background_sfs = background_sfs
        
        bins = sorted(foreground_sfs.keys())
        
        observed_counts = []
        for k in bins[1:-1]:
            count = foreground_sfs[k]
            observed_counts.append(int(count))
        # print(observed_counts)
        
        # probs from background
        probabilities_bg = []
        for k in bins[1:-1]:
            probability_bg = background_sfs[k]
            probabilities_bg.append(probability_bg)
        # print(probabilities_bg)
        
        # normalize probabilities so they can add up to 1
        total_bg = sum(probabilities_bg)
        
        probabilities_bg_norm = []
        for p in probabilities_bg:
            p_norm = p/total_bg
            probabilities_bg_norm.append(p_norm)
        # print(probabilities_bg)
        
        # probs from normalized foreground
        foreground_sfs_norm = self.normalize_1d_sfs(foreground_sfs)
        
        probabilities_fg = []
        for k in bins[1:-1]:
            probability_fg = foreground_sfs_norm[k]
            probabilities_fg.append(probability_fg)
        # print(probabilities_fg)
        
        # normalize probabilities so they can add up to 1
        total_fg = sum(probabilities_fg)
        
        probabilities_fg_norm = []
        for p in probabilities_fg:
            p_norm = p/total_fg
            probabilities_fg_norm.append(p_norm)
        # print(probabilities_fg)
        
        total_sites = sum(observed_counts) # should this exclude the counts in the first and last bin? (sum(observed_counts[1:-1]))
        # print(total_sites)
        
        log_likelihood_bg = multinomial.logpmf(x=observed_counts, n=total_sites, p=probabilities_bg_norm)
        log_likelihood_fg = multinomial.logpmf(x=observed_counts, n=total_sites, p=probabilities_fg_norm)
        
        clr = 2*(log_likelihood_fg - log_likelihood_bg)
        
        return clr 
    
    def T1D_scan(self, data_dict, background_sfs, window_size, pop, pop_size):
        
        '''
        genome scan to calculate T1D
        
        arguments: 
        - data_dict:
        - background_sfs:
        - window_size:
        '''
        
        self.data_dict = data_dict
        self.background_sfs = background_sfs # normalized SFS
        self.window_size = window_size
        self.pop = pop
        self.pop_size = pop_size
        
        # sort snps by chromosome and position
        sorted_snps = []
        for snp_key in data_dict.keys():
            coords = snp_key.split('-')
            chromosome_id = coords[0]
            position = int(coords[1])
            sorted_snps.append((chromosome_id, position, snp_key))
            
        sorted_snps.sort(key=lambda x: (x[0], x[1]))
        
        T1D_windows = {}
        current_chromosome = None
        current_window_start = 0
        window_data = {}
        
        # loop over the sorted snps in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            # reset the window start for each new chromosome
            if chrom != current_chromosome:
                if window_data:
                    foreground_sfs = self.calculate_1d_sfs(window_data, pop, pop_size, self.start_position, self.end_position, self.variant_type)
                    folded_foreground_sfs = self.fold_1d_sfs(foreground_sfs)
                    T1D = self.calculate_likelihood_1D(folded_foreground_sfs, self.background_sfs)
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    T1D_windows[window_range] = {
                        "snp_count": snp_count,
                        "T1D": T1D
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
                    foreground_sfs = self.calculate_1d_sfs(window_data, pop, pop_size, self.start_position, self.end_position, self.variant_type)
                    folded_foreground_sfs = self.fold_1d_sfs(foreground_sfs)
                    T1D = self.calculate_likelihood_1D(folded_foreground_sfs, self.background_sfs)
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    T1D_windows[window_range] = {
                        "snp_count": snp_count,
                        "T1D": T1D
                    }

                # move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}

        # calculate for the last window if it has any data
        if window_data:
            foreground_sfs = self.calculate_1d_sfs(window_data, pop, pop_size, self.start_position, self.end_position, self.variant_type)
            folded_foreground_sfs = self.fold_1d_sfs(foreground_sfs)
            T1D = self.calculate_likelihood_1D(folded_foreground_sfs, self.background_sfs)
            snp_count = self.count_snps(window_data, self.variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            T1D_windows[window_range] = {
                "snp_count": snp_count,
                "T1D": T1D
            }
            
        return T1D_windows 
    
    def calculate_likelihood_2D(self, foreground_2d_sfs, background_2d_sfs):
        
        self.foreground_2d_sfs = foreground_2d_sfs
        self.background_2d_sfs = background_2d_sfs
        
        bins = sorted(foreground_2d_sfs.keys())
        # print(bins)
        
        observed_counts = []
        for k in bins[1:-1]:
            count = foreground_2d_sfs[k]
            observed_counts.append(int(count))
        # print(observed_counts)
        
        # probs from background
        probabilities_bg = []
        for k in bins[1:-1]:
            probability_bg = background_2d_sfs[k]
            probabilities_bg.append(probability_bg)
        # print(probabilities_bg)
        
        # normalize probabilities so they can add up to 1
        total_bg = sum(probabilities_bg)
        
        probabilities_bg_norm = []
        for p in probabilities_bg:
            p_norm = p/total_bg
            probabilities_bg_norm.append(p_norm)
        # print(probabilities_bg)
        
        # probs from normalized foreground
        foreground_sfs_norm = self.normalize_2d_sfs(foreground_2d_sfs)
        
        probabilities_fg = []
        for k in bins[1:-1]:
            probability_fg = foreground_sfs_norm[k]
            probabilities_fg.append(probability_fg)
        # print(probabilities_fg)
        
        # normalize probabilities so they can add up to 1
        total_fg = sum(probabilities_fg)
        
        probabilities_fg_norm = []
        for p in probabilities_fg:
            p_norm = p/total_fg
            probabilities_fg_norm.append(p_norm)
        # print(probabilities_fg)
        
        total_sites = sum(observed_counts) # should this exclude the counts in the first and last bin? (sum(observed_counts[1:-1]))
        # print(total_sites)
        
        log_likelihood_bg = multinomial.logpmf(x=observed_counts, n=total_sites, p=probabilities_bg_norm)
        log_likelihood_fg = multinomial.logpmf(x=observed_counts, n=total_sites, p=probabilities_fg_norm)
        
        clr = 2*(log_likelihood_fg - log_likelihood_bg)
        
        return clr           
    
    def T2D_scan(self, data_dict, background_2d_sfs, window_size):
        
        '''
        genome scan to calculate T1D
        
        arguments: 
        - data_dict:
        - background_sfs:
        - window_size:
        '''
        
        self.data_dict = data_dict
        self.background_2d_sfs = background_2d_sfs
        self.window_size = window_size
        
        
        
        # sort snps by chromosome and position
        sorted_snps = []
        for snp_key in data_dict.keys():
            coords = snp_key.split('-')
            chromosome_id = coords[0]
            position = int(coords[1])
            sorted_snps.append((chromosome_id, position, snp_key))
            
        sorted_snps.sort(key=lambda x: (x[0], x[1]))
        
        T2D_windows = {}
        current_chromosome = None
        current_window_start = 0
        window_data = {}
        
        # loop over the sorted snps in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            # reset the window start for each new chromosome
            if chrom != current_chromosome:
                if window_data:
                    foreground_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(foreground_2d_sfs, self.background_2d_sfs)
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    T2D_windows[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D
                    }

                # start new chromosome
                current_chromosome = chrom
                current_window_start = 1
                window_data = {}
                
                # calculate chromosome-specific background SFS
                chromosome_snps = {}
                
                for snp_key, snp_data in data_dict.items():
                    if snp_key.startswith(f"{chrom}-"):
                        chromosome_snps[snp_key] = snp_data
                        
                background_2d_sfs = self.calculate_2d_sfs(chromosome_snps)
            
            # check if SNP is within the current window
            if pos < current_window_start + window_size:
                window_data[snp_key] = data_dict[snp_key]  # add SNP to current window
            else:
                # calculate p-values for the current window
                if window_data:
                    foreground_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(foreground_2d_sfs, self.background_2d_sfs)
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    T2D_windows[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D
                    }

                # move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}

        # calculate for the last window if it has any data
        if window_data:
            foreground_2d_sfs = self.calculate_2d_sfs(window_data)
            T2D = self.calculate_likelihood_2D(foreground_2d_sfs, self.background_2d_sfs)
            snp_count = self.count_snps(window_data, self.variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            T2D_windows[window_range] = {
                "snp_count": snp_count,
                "T2D": T2D
            }
            
        return T2D_windows  
    
    
    def new_term(self, T1D, T2D):
        self.T1D = T1D
        self.T2D = T2D
        
        jointsfs_contribution = T2D - T1D
        
        return jointsfs_contribution
    
    def combined_scan(self, data_dict, window_size):
        
        self.data_dict = data_dict
        self.window_size = window_size
        
        # precompute bg SFS for each chromosome
        bg_2d_sfs_per_chr = {}
        bg_1d_sfs_pop1_per_chr = {}
        bg_1d_sfs_pop2_per_chr = {}
        
        # group SNPs by chr
        snps_by_chr = {}
        for snp_key in data_dict.keys():
            coords = snp_key.split('-')
            chromosome_id = coords[0]
            if chromosome_id not in snps_by_chr:
                snps_by_chr[chromosome_id] = {}
            snps_by_chr[chromosome_id][snp_key] = data_dict[snp_key]
            
        # calculate bg SFS for each chromosome
        for chrom, snp_data in snps_by_chr.items():
            # calculate background 2d sfs
            bg_2d_sfs = self.calculate_2d_sfs(snp_data)
            bg_2d_sfs_per_chr[chrom] = self.normalize_2d_sfs(bg_2d_sfs)
            
            # calculate background 1d sfs for pop1, then fold it, and then normalize it
            bg_1d_sfs_pop1 = self.calculate_1d_sfs(snp_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
            bg_1d_sfs_pop1 = self.fold_1d_sfs(bg_1d_sfs_pop1)
            bg_1d_sfs_pop1_per_chr[chrom] = self.normalize_1d_sfs(bg_1d_sfs_pop1)
            
            # calculate background 1d sfs for pop2, then fold it, and then normalize it
            bg_1d_sfs_pop2 = self.calculate_1d_sfs(snp_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
            bg_1d_sfs_pop2 = self.fold_1d_sfs(bg_1d_sfs_pop2)
            bg_1d_sfs_pop2_per_chr[chrom] = self.normalize_1d_sfs(bg_1d_sfs_pop2)
            
        # sort SNPs by chromosome and position
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
        
        # loop over the sorted SNPs in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            if chrom != current_chromosome:
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs_per_chr[current_chromosome])
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1_per_chr[current_chromosome])
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2_per_chr[current_chromosome])
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # start new chromosome
                current_chromosome = chrom
                current_window_start = 1
                window_data = {}
                
            # check if SNP is within the current window
            if pos < current_window_start + window_size:
                window_data[snp_key] = data_dict[snp_key]
            else:
                # calculate stats for current window
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs_per_chr[current_chromosome])
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1_per_chr[current_chromosome])
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2_per_chr[current_chromosome])
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # Move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}
                
        # calculate stats for last window
        if window_data:
            # calculate T2D
            fg_2d_sfs = self.calculate_2d_sfs(window_data)
            T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs_per_chr[current_chromosome])
            
            # calculate T1D for pop1
            fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
            T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1_per_chr[current_chromosome])
            
            # calculate T1D for pop2
            fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
            T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2_per_chr[current_chromosome])
            
            # calculate new terms
            new_term_pop1 = T2D - T1D_pop1
            new_term_pop2 = T2D - T1D_pop2
            
            snp_count = self.count_snps(window_data, self.variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            results[window_range] = {
                "snp_count": snp_count,
                "T2D": T2D,
                "T1D_pop1": T1D_pop1,
                "T1D_pop2": T1D_pop2,
                "new_term_pop1": new_term_pop1,
                "new_term_pop2": new_term_pop2
            }
        
        return results
    
    def scan_chooseChr(self, data_dict, window_size, background_chromosome):
        '''
        Genome scan to calculate T2D, T1D for two populations, and new terms (T2D - T1D) for each population.
        Uses a specified chromosome as the background for all calculations.
        
        Arguments: 
        - data_dict: Dictionary containing SNP data.
        - window_size: Size of the window for scanning.
        - background_chromosome: Chromosome to use as the background for SFS calculations.
        '''
        
        self.data_dict = data_dict
        self.window_size = window_size
        
        # group SNPs by chromosome
        snps_by_chr = {}
        for snp_key in data_dict.keys():
            coords = snp_key.split('-')
            chromosome_id = coords[0]
            if chromosome_id not in snps_by_chr:
                snps_by_chr[chromosome_id] = {}
            snps_by_chr[chromosome_id][snp_key] = data_dict[snp_key]
        
        # check if the specified background chromosome exists in the data
        if background_chromosome not in snps_by_chr:
            raise ValueError(f"Background chromosome {background_chromosome} not found in the data.")
        
        # calculate background SFS for the specified chromosome
        bg_snp_data = snps_by_chr[background_chromosome]
        
        # calculate background 2d sfs and normalize
        bg_2d_sfs = self.calculate_2d_sfs(bg_snp_data)
        bg_2d_sfs = self.normalize_2d_sfs(bg_2d_sfs)
        
        # calculate background 1d sfs for pop1, fold, and normalize
        bg_1d_sfs_pop1 = self.calculate_1d_sfs(bg_snp_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
        bg_1d_sfs_pop1 = self.fold_1d_sfs(bg_1d_sfs_pop1)
        bg_1d_sfs_pop1 = self.normalize_1d_sfs(bg_1d_sfs_pop1)
        
        # calculate background 1d sfs for pop2, fold, and normalize
        bg_1d_sfs_pop2 = self.calculate_1d_sfs(bg_snp_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
        bg_1d_sfs_pop2 = self.fold_1d_sfs(bg_1d_sfs_pop2)
        bg_1d_sfs_pop2 = self.normalize_1d_sfs(bg_1d_sfs_pop2)
        
        # sort SNPs by chromosome and position
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
        
        # loop over the sorted SNPs in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            if chrom != current_chromosome:
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # start new chromosome
                current_chromosome = chrom
                current_window_start = 1
                window_data = {}
                
            # check if SNP is within the current window
            if pos < current_window_start + window_size:
                window_data[snp_key] = data_dict[snp_key]
            else:
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # Move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}
                
        # calculate stats for last window
        if window_data:
            # calculate T2D
            fg_2d_sfs = self.calculate_2d_sfs(window_data)
            T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
            
            # calculate T1D for pop1
            fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
            T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
            
            # calculate T1D for pop2
            fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
            T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
            
            # calculate new terms
            new_term_pop1 = T2D - T1D_pop1
            new_term_pop2 = T2D - T1D_pop2
            
            snp_count = self.count_snps(window_data, self.variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            results[window_range] = {
                "snp_count": snp_count,
                "T2D": T2D,
                "T1D_pop1": T1D_pop1,
                "T1D_pop2": T1D_pop2,
                "new_term_pop1": new_term_pop1,
                "new_term_pop2": new_term_pop2
            }
        
        return results
    
    def scan_precomputed_BG(self, data_dict, window_size, bg_2d_sfs, bg_1d_sfs_pop1, bg_1d_sfs_pop2):
        '''
        Genome scan to calculate T2D, T1D for two populations, and new terms (T2D - T1D) for each population.
        Uses precomputed background SFS for all calculations.
        
        Arguments: 
        - data_dict: Dictionary containing SNP data.
        - window_size: Size of the window for scanning.
        - bg_2d_sfs: Precomputed and normalized 2D SFS to use as the background.
        - bg_1d_sfs_pop1: Precomputed and normalized 1D SFS for population 1 to use as the background.
        - bg_1d_sfs_pop2: Precomputed and normalized 1D SFS for population 2 to use as the background.
        '''
        
        self.data_dict = data_dict
        self.window_size = window_size
        
        # sort SNPs by chromosome and position
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
        
        # loop over the sorted SNPs in windows of window_size
        for chrom, pos, snp_key in sorted_snps:
            if chrom != current_chromosome:
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # start new chromosome
                current_chromosome = chrom
                current_window_start = 1
                window_data = {}
                
            # check if SNP is within the current window
            if pos < current_window_start + window_size:
                window_data[snp_key] = data_dict[snp_key]
            else:
                if window_data:
                    # calculate T2D
                    fg_2d_sfs = self.calculate_2d_sfs(window_data)
                    T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
                    
                    # calculate T1D for pop1
                    fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
                    T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
                    
                    # calculate T1D for pop2
                    fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
                    folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
                    T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
                    
                    # calculate new terms
                    new_term_pop1 = T2D - T1D_pop1
                    new_term_pop2 = T2D - T1D_pop2
                    
                    snp_count = self.count_snps(window_data, self.variant_type)
                    window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
                    results[window_range] = {
                        "snp_count": snp_count,
                        "T2D": T2D,
                        "T1D_pop1": T1D_pop1,
                        "T1D_pop2": T1D_pop2,
                        "new_term_pop1": new_term_pop1,
                        "new_term_pop2": new_term_pop2
                    }
                    
                # Move to the next window, aligned to the window size
                current_window_start += window_size * ((pos - current_window_start) // window_size)
                window_data = {snp_key: data_dict[snp_key]}
                
        # calculate stats for last window
        if window_data:
            # calculate T2D
            fg_2d_sfs = self.calculate_2d_sfs(window_data)
            T2D = self.calculate_likelihood_2D(fg_2d_sfs, bg_2d_sfs)
            
            # calculate T1D for pop1
            fg_sfs_pop1 = self.calculate_1d_sfs(window_data, self.pop1, self.pop1_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop1 = self.fold_1d_sfs(fg_sfs_pop1)
            T1D_pop1 = self.calculate_likelihood_1D(folded_fg_sfs_pop1, bg_1d_sfs_pop1)
            
            # calculate T1D for pop2
            fg_sfs_pop2 = self.calculate_1d_sfs(window_data, self.pop2, self.pop2_size, self.start_position, self.end_position, self.variant_type)
            folded_fg_sfs_pop2 = self.fold_1d_sfs(fg_sfs_pop2)
            T1D_pop2 = self.calculate_likelihood_1D(folded_fg_sfs_pop2, bg_1d_sfs_pop2)
            
            # calculate new terms
            new_term_pop1 = T2D - T1D_pop1
            new_term_pop2 = T2D - T1D_pop2
            
            snp_count = self.count_snps(window_data, self.variant_type)
            window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
            results[window_range] = {
                "snp_count": snp_count,
                "T2D": T2D,
                "T1D_pop1": T1D_pop1,
                "T1D_pop2": T1D_pop2,
                "new_term_pop1": new_term_pop1,
                "new_term_pop2": new_term_pop2
            }
        
        return results                
    
    def get_gens(self, main_dir):
        search_strings = set()
        for root, dirs, files in os.walk(main_dir):
            for file in files:
                parts = file.split('.')
                if len(parts) == 5:
                    search_strings.add(parts[1])
        return search_strings
    
    def scan_sims(self, main_dir):
        
        # define pop map file
        popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

        # get generation IDs
        generations = self.get_gens(main_dir)
        
        results = {}
        
        # get list of target vcfs and concatenated vcf files
        for generation in generations:
            target_vcfs = glob.glob(f"{main_dir}/iter*/*{generation}*.vcf.gz")
            concatenated_vcfs = glob.glob(f"{main_dir}/concatenated_vcfs/gen.{generation}.concatenated.vcf.gz")
            
            # get background sfs from concatenated vcfs
            for vcf in concatenated_vcfs:
                data_dict = self.make_data_dict_vcf(vcf, popinfo_filename, snp_type=None)
                background_sfs = self.calculate_2d_sfs(data_dict, 'p1', 'p2', start_position=0, end_position=500000)
                normalized_background_sfs = self.normalize_2d_sfs(background_sfs)
                
                for vcf_input in target_vcfs:
                    data_dict_target = self.make_data_dict_vcf(vcf_input, popinfo_filename, snp_type=None)
                    p_values_sims = calculate_p_window(data_dict_target, normalized_background_sfs, 500000, 'p1', 'p2', start_position=None, end_position=None)
                    # print(p_values_sims)
                    
                    # extract iteration number from filename
                    iteration_number = int(vcf_input.split('.')[2])
                
                    for key, value in p_values_sims.items():
                        # Determine region based on window coordinates
                        window_start, window_end = key.split('_')[1].split('-')
                        region = 'background' if int(window_end) <= 1000000 else 'foreground'
        
                        writer.writerow({
                            'generation': generation,
                            'iteration': iteration_number,
                            'region': region,
                            'window_coords': key,
                            'likelihood': value
                        })
        
        
    

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

chromosome_ids = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/scripts/chromosomes.txt"

chr_ids_file = open(chromosome_ids, "r")
chr_ids = {}

for line in chr_ids_file:
    columns = line.strip().split("\t")
    if len(columns) >= 2:
        chr_ids[columns[0]] = columns[1]
chr_ids_file.close()

    
def plot_manhattan(T1D_windows, chr_mapping, stat, title, threshold=None, ylim=None):
    """
    Generate a Manhattan plot for likelihood values across the genome.
    
    Arguments:
    - T1D_windows: Dictionary where keys are window ranges (e.g., "NC_087088.1 1000-2000"),
      and values are dictionaries containing 'T1D' likelihood values.
    - chr_mapping: Dictionary mapping chromosome accession IDs to numeric values.
    - stat: The statistic to plot (e.g., 'T1D', 'T2D', etc.).
    - title: Title of the plot.
    - threshold: Optional; If provided, highlights the top percentage of windows above this threshold.
    - ylim: Optional; Tuple (ymin, ymax) to set the limits of the y-axis.
    """
    
    # Extract chromosome, position, and likelihood values
    chroms, positions, likelihoods = [], [], []
    
    for window, values in T1D_windows.items():
        chrom, pos_range = window.split()
        start_pos = int(pos_range.split('-')[0])
        
        if chrom in chr_mapping:
            chrom_num = chr_mapping[chrom]
            chroms.append(chrom_num)
            positions.append(start_pos)
            likelihoods.append(values[stat])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({'chromosome': chroms, 'position': positions, 'likelihood': likelihoods})
    df['chromosome'] = df['chromosome'].astype('category')
    df['chromosome'] = df['chromosome'].cat.set_categories(sorted(df['chromosome'].unique(), key=int), ordered=True)
    df = df.sort_values(['chromosome', 'position'])
    df['ind'] = range(len(df))
    
    # Compute threshold if specified
    highlight = pd.Series(False, index=df.index)
    if threshold is not None:
        threshold_value = np.percentile(df['likelihood'], 100 - threshold)
        highlight = df['likelihood'] >= threshold_value
    
    # Group by chromosome
    df_grouped = df.groupby('chromosome')
    
    # Create Manhattan plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['navy', 'lightskyblue']
    x_labels = []
    x_labels_pos = []
    
    for i, (name, group) in enumerate(df_grouped):
        color = colors[i % len(colors)]
        ax.scatter(group['ind'], group['likelihood'], c=color, s=15, alpha=0.7)
    
    # Highlight significant points
    if threshold is not None:
        ax.scatter(df.loc[highlight, 'ind'], df.loc[highlight, 'likelihood'], c='salmon', s=15, edgecolors='red')
    
    # Plot significance threshold line
    if threshold is not None:
        ax.axhline(y=threshold_value, color='black', linestyle='--', linewidth=1)
    
    for i, (name, group) in enumerate(df_grouped):
        if i % 2 == 0:  # Show label only for every other chromosome
            x_labels.append(name)
            x_labels_pos.append((group['ind'].iloc[-1] + group['ind'].iloc[0]) / 2)
    
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("CLR")
    ax.set_title(title)
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.show()
    

''' chr1 files '''
chr1_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchr1.vcf.gz"
popmap = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/popmap.txt"


inferencePipeline = LikelihoodInference_jointSFS(chr1_vcf, popmap)

chr1_data_dict = inferencePipeline.make_data_dict_vcf()

#store chr1 SNP dictionary
# with bz2.BZ2File('chr1.pkl.bz2', 'wb') as file:
#     pickle.dump(chr1_data_dict, file)

#load chr1 SNP dictionary
with bz2.BZ2File('chr1.pkl.bz2', 'rb') as file:
    chr1_data_dict = pickle.load(file)

# calculate folded sfs
chr1_2d_sfs = inferencePipeline.calculate_2d_sfs(chr1_data_dict)
plot_2d_sfs(chr1_2d_sfs, (36, 28), pop_ids=('uv', 'bv'))

chr1_2d_sfs_norm = inferencePipeline.normalize_2d_sfs(chr1_2d_sfs)


# uv
chr1_uv_sfs = inferencePipeline.calculate_1d_sfs(chr1_data_dict, 'uv', 18, start_position=None, end_position=None, variant_type=None)
chr1_uv_sfs_folded = inferencePipeline.fold_1d_sfs(chr1_uv_sfs)
chr1_uv_norm = inferencePipeline.normalize_1d_sfs(chr1_uv_sfs_folded)

# bv
chr1_bv_sfs = inferencePipeline.calculate_1d_sfs(chr1_data_dict, 'bv', 14, start_position=None, end_position=None, variant_type=None)
chr1_bv_sfs_folded = inferencePipeline.fold_1d_sfs(chr1_bv_sfs)
chr1_bv_norm = inferencePipeline.normalize_1d_sfs(chr1_bv_sfs_folded)


''' chrZ window with highest FST '''
chrZ_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchrZ_highestFSTwindow.vcf.gz"
inference_chrZ = LikelihoodInference_jointSFS(chrZ_vcf, popmap)

chrZ_dict = inference_chrZ.make_data_dict_vcf()

#uv
chrZ_uv_sfs = inference_chrZ.calculate_1d_sfs(chrZ_dict, 'uv', 18, start_position=None, end_position=None, variant_type=None)
chrZ_uv_sfs_folded = inference_chrZ.fold_1d_sfs(chrZ_uv_sfs)
chrZ_uv_norm = inference_chrZ.normalize_1d_sfs(chrZ_uv_sfs_folded)

#bv
chrZ_bv_sfs = inference_chrZ.calculate_1d_sfs(chrZ_dict, 'bv', 14, start_position=None, end_position=None, variant_type=None)
chrZ_bv_sfs_folded = inference_chrZ.fold_1d_sfs(chrZ_bv_sfs)
chrZ_bv_norm = inference_chrZ.normalize_1d_sfs(chrZ_bv_sfs_folded)

chrZ_2d_sfs = inference_chrZ.calculate_2d_sfs(chrZ_dict)
plot_2d_sfs(chrZ_2d_sfs, (36, 28), pop_ids=('uv', 'bv'))


# calculate 1D likelihoods
chr1_likelihoods = inferencePipeline.calculate_likelihood_1D(chr1_uv_sfs_folded, chr1_uv_norm)
uv_chr1_chrZ_clr = inferencePipeline.calculate_likelihood_1D(chrZ_uv_sfs_folded, chr1_uv_norm)
bv_chr1_chrZ_clr = inferencePipeline.calculate_likelihood_1D(chrZ_bv_sfs_folded, chr1_bv_norm)

# calculate 2D likelihoods conditional on 1D
chr1_chrZ_T2D = inferencePipeline.calculate_likelihood_2D(chrZ_2d_sfs, chr1_2d_sfs, chrZ_uv_sfs_folded, chr1_uv_norm)

''' plot CLRs - T1D and T2D '''

# load compressed data
with bz2.BZ2File('../data_summer2024/likelihood_scan/genome_data.pkl.bz2', 'rb') as file:
    ECB_wg_dict = pickle.load(file)

ECB_uv_T1D = inferencePipeline.T1D_scan(ECB_wg_dict, chr1_uv_norm, 500000, "uv", 18)
plot_manhattan(ECB_uv_T1D, chr_ids, 'T1D', "univoltine - 500kb windows - excluding fixed sites", threshold=5)

ECB_bv_T1D = inferencePipeline.T1D_scan(ECB_wg_dict, chr1_bv_norm, 500000, "bv", 14)
plot_manhattan(ECB_bv_T1D, chr_ids, 'T1D', "bivoltine - 500kb windows - excluding fixed sites", threshold=5)

ECB_T2D = inferencePipeline.T2D_scan(ECB_wg_dict, chr1_2d_sfs_norm, 100000)
plot_manhattan(ECB_T2D, chr_ids, 'T2D', "T2D - 100kb windows", threshold=5)

''' each chromosome as its own background '''
ECB_stats_500kb = inferencePipeline.combined_scan(ECB_wg_dict, 500000)
plot_manhattan(ECB_stats_500kb, chr_ids, 'T1D_pop1', "univoltine T1D - 500kb windows", ylim=(0,26000))
plot_manhattan(ECB_stats_500kb, chr_ids, 'T1D_pop2', "bivoltine T1D - 500kb windows", ylim=(0,26000))
plot_manhattan(ECB_stats_500kb, chr_ids, 'T2D', "T2D - 500kb windows", threshold=5, ylim=(0,26000))
plot_manhattan(ECB_stats_500kb, chr_ids, 'new_term_pop1', "univoltine new_term - 500kb windows", ylim=(0,26000))
plot_manhattan(ECB_stats_500kb, chr_ids, 'new_term_pop2', "bivoltine new_term - 500kb windows", ylim=(0,26000))

ECB_stats_100kb = inferencePipeline.combined_scan(ECB_wg_dict, 100000)
plot_manhattan(ECB_stats_100kb, chr_ids, 'T1D_pop1', "univoltine T1D - 100kb windows")
plot_manhattan(ECB_stats_100kb, chr_ids, 'T1D_pop2', "bivoltine T1D - 100kb windows")
plot_manhattan(ECB_stats_100kb, chr_ids, 'T2D', "T2D - 100kb windows", ylim=(0,26000))
plot_manhattan(ECB_stats_100kb, chr_ids, 'new_term_pop1', "univoltine new_term - 100kb windows")
plot_manhattan(ECB_stats_100kb, chr_ids, 'new_term_pop2', "bivoltine new_term - 100kb windows")

''' specifying which chromosome to use as background '''
ECB_bgchr1_500kb = inferencePipeline.scan_chooseChr(ECB_wg_dict, 500000, 'NC_087088.1')
plot_manhattan(ECB_bgchr1_500kb, chr_ids, 'T1D_pop1', "univoltine T1D - chr1 background - 500kb windows")
plot_manhattan(ECB_bgchr1_500kb, chr_ids, 'T1D_pop2', "bivoltine T1D - chr1 background - 500kb windows")
plot_manhattan(ECB_bgchr1_500kb, chr_ids, 'T2D', "T2D - chr1 background - 500kb windows", threshold=5, ylim=(0,26000))
plot_manhattan(ECB_bgchr1_500kb, chr_ids, 'new_term_pop1', "univoltine new_term - chr1 background - 500kb windows")
plot_manhattan(ECB_bgchr1_500kb, chr_ids, 'new_term_pop2', "bivoltine new_term - chr1 background - 500kb windows")

ECB_bgchr1_100kb = inferencePipeline.scan_chooseChr(ECB_wg_dict, 100000, 'NC_087088.1')
plot_manhattan(ECB_bgchr1_100kb, chr_ids, 'T1D_pop1', "univoltine T1D - chr1 background - 100kb windows")
plot_manhattan(ECB_bgchr1_100kb, chr_ids, 'T1D_pop2', "bivoltine T1D - chr1 background - 100kb windows")
plot_manhattan(ECB_bgchr1_100kb, chr_ids, 'T2D', "T2D - chr1 background - 100kb windows")
plot_manhattan(ECB_bgchr1_100kb, chr_ids, 'new_term_pop1', "univoltine new_term - chr1 background - 100kb windows")
plot_manhattan(ECB_bgchr1_100kb, chr_ids, 'new_term_pop2', "bivoltine new_term - chr1 background - 100kb windows")

''' use whole genome SFS as background '''
ECB_2d_sfs = inferencePipeline.calculate_2d_sfs(ECB_wg_dict)
ECB_2d_sfs = inferencePipeline.normalize_2d_sfs(ECB_2d_sfs)

ECB_uv_sfs = inferencePipeline.calculate_1d_sfs(ECB_wg_dict, 'uv', 18, start_position=None, end_position=None, variant_type=None)
ECB_uv_sfs = inferencePipeline.fold_1d_sfs(ECB_uv_sfs)
ECB_uv_sfs = inferencePipeline.normalize_1d_sfs(ECB_uv_sfs)

ECB_bv_sfs = inferencePipeline.calculate_1d_sfs(ECB_wg_dict, 'bv', 14, start_position=None, end_position=None, variant_type=None)
ECB_bv_sfs = inferencePipeline.fold_1d_sfs(ECB_bv_sfs)
ECB_bv_sfs = inferencePipeline.normalize_1d_sfs(ECB_bv_sfs)

ECB_bgWG_500kb = inferencePipeline.scan_precomputed_BG(ECB_wg_dict, 500000, ECB_2d_sfs, ECB_uv_sfs, ECB_bv_sfs)
plot_manhattan(ECB_bgWG_500kb, chr_ids, 'T1D_pop1', "univoltine T1D - whole genome background - 500kb windows")
plot_manhattan(ECB_bgWG_500kb, chr_ids, 'T1D_pop2', "bivoltine T1D - whole genome background - 500kb windows")
plot_manhattan(ECB_bgWG_500kb, chr_ids, 'T2D', "T2D - whole genome background - 500kb windows", threshold=5, ylim=(0,26000))
plot_manhattan(ECB_bgWG_500kb, chr_ids, 'new_term_pop1', "univoltine new_term - whole genome background - 500kb windows")
plot_manhattan(ECB_bgWG_500kb, chr_ids, 'new_term_pop2', "bivoltine new_term - whole genome background - 500kb windows")

ECB_bgWG_100kb = inferencePipeline.scan_precomputed_BG(ECB_wg_dict, 100000, ECB_2d_sfs, ECB_uv_sfs, ECB_bv_sfs)
plot_manhattan(ECB_bgWG_100kb, chr_ids, 'T1D_pop1', "univoltine T1D - whole genome background - 100kb windows")
plot_manhattan(ECB_bgWG_100kb, chr_ids, 'T1D_pop2', "bivoltine T1D - whole genome background - 100kb windows")
plot_manhattan(ECB_bgWG_100kb, chr_ids, 'T2D', "T2D - whole genome background - 100kb windows", ylim=(0,26000))
plot_manhattan(ECB_bgWG_100kb, chr_ids, 'new_term_pop1', "univoltine new_term - whole genome background - 100kb windows")
plot_manhattan(ECB_bgWG_100kb, chr_ids, 'new_term_pop2', "bivoltine new_term - whole genome background - 100kb windows")












