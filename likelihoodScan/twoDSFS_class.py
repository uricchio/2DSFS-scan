import gzip
import re
import math
from scipy.stats import poisson
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
    # def __init__(self, vcf_filename, popinfo_filename, data_dict, pop1, pop2, pop1_size, pop2_size, ):
    #     self.vcf_filename = vcf_filename 
    #     self.popinfo_filename = popinfo_filename
    #     self.data_dict = data_dict
    #     self.pop1 = pop1
    #     self.pop2 = pop2
    #     self.pop1_size = pop1_size
    #     self.pop2_size = pop2_size
        
    
    def make_data_dict_vcf(self, vcf_filename, popinfo_filename):
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
        
        self.vcf_filename = vcf_filename
        self.popinfo_filename = popinfo_filename
        
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


    def calculate_2d_sfs(self, data_dict, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type=None):
        """
        calculate the two-dimensional site frequency spectrum (SFS) 
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
        self.pop1 = pop1
        self.pop2 = pop2
        self.pop1_size = pop1_size
        self.pop2_size = pop2_size
        self.start_position = start_position
        self.end_position = end_position
        self.variant_type = variant_type

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

            alt_count_pop1 = pop1_calls[1]
            alt_count_pop2 = pop2_calls[1]

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
        counts = list(sfs.values())
        total = sum(counts[1:-1]) # exclude first and last bin 
        
        # divide each bin value by the total number of sites
        normalized_sfs = {}
        for coords, values in sfs.items():
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
    
    def fold_2d_sfs(self, sfs_dict, pop1_size, pop2_size):
        num_chromosomes_pop1 = pop1_size * 2
        num_chromosomes_pop2 = pop2_size * 2
    
        folded_sfs_dict = {}
    
        for (freq_pop1, freq_pop2), count in sfs_dict.items():
            maf_pop1 = min(freq_pop1, num_chromosomes_pop1 - freq_pop1) # freq of alt, freq of ref (# chr - freq of alt)
            maf_pop2 = min(freq_pop2, num_chromosomes_pop2 - freq_pop2)
            
            key = (maf_pop1, maf_pop2)
            
            # initialize if key does not exist
            if key not in folded_sfs_dict:
                folded_sfs_dict[key] = 0
                
            folded_sfs_dict[key] += count
        
        return folded_sfs_dict
    
    
# chr1 files
chr1_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchr1.vcf.gz"
popmap = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/popmap.txt"

inferencePipeline = LikelihoodInference_jointSFS()

chr1_data_dict = inferencePipeline.make_data_dict_vcf(chr1_vcf, popmap)
#store chr1 SNP dictionary
# with bz2.BZ2File('chr1.pkl.bz2', 'wb') as file:
#     pickle.dump(chr1_data_dict, file)

#load chr1 SNP dictionary
with bz2.BZ2File('chr1.pkl.bz2', 'rb') as file:
    chr1_data_dict = pickle.load(file)

# calculate unfolded sfs
chr1_2d_sfs = inferencePipeline.calculate_2d_sfs(chr1_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type=None)

# fold sfs
chr1_2d_sfs_folded = inferencePipeline.fold_2d_sfs(chr1_2d_sfs, 18, 14)

        