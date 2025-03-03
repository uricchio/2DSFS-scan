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


''' parsing simulation vcfs from multiple runs'''

# 1. get search strings for each generation
main_dir = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/test"

def get_gens(main_dir):
    search_strings = set()
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            parts = file.split('.')
            if len(parts) == 5:
                search_strings.add(parts[1])
    return search_strings

def concatenate_vcf_files(main_dir):
    
    search_strings = get_gens(main_dir)
    
    for pattern in search_strings:
        vcf_files = glob.glob(f"{main_dir}/*/*{pattern}*.vcf*")
        first_vcf = vcf_files[0]
        
        with open(first_vcf, 'r') as f:
            header_lines = []
            for line in f:
                if line.startswith('##') or line.startswith('#'):
                    header_lines.append(line)
        
        output_file = f"{main_dir}/concatenated_vcfs/gen.{pattern}.concatenated.vcf"
        with open(output_file, 'w') as out:
            out.writelines(header_lines)
        
            for file in vcf_files:
                with open(file, 'r') as f:
                    for line in f:
                        if not line.startswith('##') and not line.startswith('#'):
                            out.write(line)
            
        print(f"VCF files containing '{pattern}' have been concatenated into {output_file}")

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
    

path = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs"

concatenate_fst_files(path)



'''simulations data'''
vcf_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/stabSel_sim.vcf.gz"
popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims.txt"


"check the software license and see if it can be reused for other purposes"

''' 1) get allele frequencies from vcf - actually, it should just generate the 2D-SFS. Take the VCF and the popmap and output 
    a dictionary that has touples as keys. The keys would correspond to specific frequency of a given allele in both populations (for example: 
    (0,0), (0,1), (1,0), etc.). After this is created, have the function take the background genome and generate its 2D-SFS and normalize it
        by summing all the sites and divide each bin in the SFS by the total number of segregating sites.'''


def make_data_dict_vcf(vcf_filename, popinfo_filename, filter=True):
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
        # print(annotation)
        
        
        # if snp_type:
        #     if not re.search(snp_type, info_field, re.IGNORECASE):
        #         continue  # skip if snp_type does not match the annotation
        #     # print(info_field)
        
        # skip snp if filter is not PASS
        if filter and cols[6] != 'PASS' and cols[6] != '.':
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

def calculate_2d_sfs(data_dict, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type=None):
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
    
    # # count number of genomes (diploid individuals)
    # num_genomes_p1 = 0
    # num_genomes_p2 = 0

    # for snp_id, snp_info in data_dict.items():    
    #     if pop1 in snp_info['calls']:
    #         num_genomes_p1 = sum(snp_info['calls'][pop1])
    #     if pop2 in snp_info['calls']:
    #         num_genomes_p2 = sum(snp_info['calls'][pop2])
    #     break

    num_genomes_p1 = pop1_size*2
    num_genomes_p2 = pop2_size*2
    
    
    # intialize 2d-sfs with keys for all combinations of frequencies    
    sfs_dict = {}
    
    # print("Number of genomes in uv:", num_genomes_p1)
    # print("Number of genomes in bv:", num_genomes_p2)
    
    for i in range(num_genomes_p1 + 1):
        for j in range(num_genomes_p2 + 1):
            sfs_dict[(i,j)] = 0 
    
    # initialize variable for total number of sites
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
            
        #     # print(f"SNP {snp_id} annotation: {snp_annotation}")
        # if snp_annotation != variant_type:
        #     continue
        
        # print(snp_annotation)
        
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
    
# wrapper function takes the 2d-sfs and normalizes the windows (it also would need the start and stop) and calculates the probability

''' 2) make 2D-SFS and plot - actually, this should be a wrapper function that uses the previous function and
    extracts the 2D-SFS on a window basis on the foreground genome, and it would have a start and a stop argument
    based on the coordinates. Then the function should calculate P(D|M), the probability of the data given the model.
    M is:
        a. calculate the total number of sites in the window across all bins (S_w)
        b. multiply S_w by the normalized SFS to get the expected number of sites (M)
        c. calculate poisson.pmf(k=S_{w,ij}, mu=M) -> S_{w,ij} is the number of sites in frequency bin i,j (allele count
           in pop i and pop j
        d. calculate the sum of the probabilities (add up log(poisson.pmf(k=S_{w,ij}, mu=M) for all values of i,j). I think
            this can be done in the third function, or wherever it makes sense!'''    

# add a key that lets me cross reference. it can be a touple: for example (0,0), as i do the count 
# add something that lets me subset the spectra (to check if it's missense or synonymous)
# add a wrapper function that does the subsetting and takes the 2d-sfs

def normalize_2d_sfs(sfs):
    
    # sum all the sites
    counts = list(sfs.values())
    total = sum(counts[1:-1]) # exclude first and last bin 
    
    # divide each bin value by the total number of sites
    normalized_sfs = {}
    for coords, values in sfs.items():
        normalized_sfs[coords] = values / total
    return normalized_sfs

def calculate_p(foreground_sfs, background_sfs):
    
    '''
    calculate the probability mass function (p values) of the foreground SFS 
    given the background SFS using poisson distribution.
    
    arguments:
    - foreground_sfs: dictionary containing the sfs of a genomic region
    - background_sfs: dictionary containing the normalized sfs of a genomic region
    '''
    
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

def count_snps(window_data, variant_type):
    snp_count = 0
    for snp_data in window_data.values():
        if variant_type is None:
            snp_count += 1
        elif snp_data.get("annotation") == variant_type:
            snp_count += 1
    return snp_count

def calculate_p_window(data_dict, sfs_normalized, window_size, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type):
    
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
                sfs_dict = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                p_values_dict = calculate_p(sfs_dict, sfs_normalized)
                snp_count = count_snps(window_data, variant_type)
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
                sfs_dict = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
                p_values_dict = calculate_p(sfs_dict, sfs_normalized)
                snp_count = count_snps(window_data, variant_type)
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
        sfs_dict = calculate_2d_sfs(window_data, pop1, pop2, pop1_size, pop2_size, start_position, end_position, variant_type)
        p_values_dict = calculate_p(sfs_dict, sfs_normalized)
        snp_count = count_snps(window_data, variant_type)
        window_range = f"{current_chromosome} {current_window_start}-{current_window_start + window_size - 1}"
        window_p_values[window_range] = {
                    "p_values": p_values_dict,
                    "snp_count": snp_count
                }
        
    return window_p_values


''' trying pipeline on ECB data '''

chrZ_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchrZ_highestFSTwindow.vcf.gz"
chr1_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchr1.vcf.gz"
chr2_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBchr2.vcf.gz"
ECB_wg_vcf = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/ECBAnnotated.vcf.gz"


popmap = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/popmap.txt"

# chr 1 - get background
chr1_data_dict = make_data_dict_vcf(chr1_vcf, popmap)
chr1_sfs = calculate_2d_sfs(chr1_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type=None) 
chr1_norm_sfs = normalize_2d_sfs(chr1_sfs)

chr1_sfs_syn = calculate_2d_sfs(chr1_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='synonymous_variant')
chr1_norm_sfs_syn = normalize_2d_sfs(chr1_sfs_syn)

chr1_sfs_nonsyn = calculate_2d_sfs(chr1_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='missense_variant')
chr1_norm_sfs_nonsyn = normalize_2d_sfs(chr1_sfs_nonsyn)

# chr Z
chrZ_data_dict = make_data_dict_vcf(chrZ_vcf, popmap)
chrZ_sfs = calculate_2d_sfs(chrZ_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type=None)
chrZ_syn_sfs = calculate_2d_sfs(chrZ_data_dict, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='synonymous_variant')

likelihood_chr1_chrZ = calculate_p_window(chrZ_data_dict, chr1_norm_sfs, 500000, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type=None)
likelihood_chr1_chrZ_syn = calculate_p_window(chrZ_data_dict, chr1_norm_sfs_syn, 100000, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='synonymous_variant')
likelihood_chr1_chrZ_nonsyn = calculate_p_window(chrZ_data_dict, chr1_norm_sfs_nonsyn, 500000, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='missense_variant')


likelihood_chr1 = calculate_p_window(chr1_data_dict, chr1_norm_sfs, 100000, 'uv', 'bv', start_position=None, end_position=None)
likelihood_chr1_syn = calculate_p_window(chr1_data_dict, chr1_norm_sfs_syn, 100000, 'uv', 'bv', 18, 14, start_position=None, end_position=None, variant_type='synonymous_variant')


# whole genome
ECB_wg_dict = make_data_dict_vcf(ECB_wg_vcf, popmap)

# store ECB_wg_dict into local machine as a compressed file so I dont have to load it everytime i need it 
with bz2.BZ2File('genome_data.pkl.bz2', 'wb') as file:
    pickle.dump(ECB_wg_dict, file)
    
# load compressed data
with bz2.BZ2File('genome_data.pkl.bz2', 'rb') as file:
    ECB_wg_dict = pickle.load(file)


# get chromosome ids

chromosome_ids = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/scripts/chromosomes.txt"

chr_ids_file = open(chromosome_ids, "r")
chr_ids = {}

for line in chr_ids_file:
    columns = line.strip().split("\t")
    if len(columns) >= 2:
        chr_ids[columns[0]] = columns[1]
chr_ids_file.close()



def write_output(output_file, background_sfs, window_size, start_position, end_position, variant_type):
    
    col_names = ['chromosome', 'region', 'window_id', 'window_start', 'window_end', 'snp_count', 'likelihood']

    with open(output_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col_names)
        writer.writeheader()

        likelihoods_wg = calculate_p_window(ECB_wg_dict, background_sfs, window_size, 'uv', 'bv', 18, 14, start_position=start_position, end_position=end_position, variant_type=variant_type)

        for key, value in likelihoods_wg.items():
            chromosome = key.split(' ')[0]

            chromosome_num = chr_ids.get(chromosome, chromosome)  # default to the original if not found
            
            region = 'foreground'

            if chromosome == 'NC_087088.1':
                region = 'background'

            window_start, window_end = key.split(' ')[1].split('-')
            
            snp_count = value['snp_count']  
            likelihood = value['p_values']  


            writer.writerow({
                'chromosome': chromosome_num,
                'region': region,
                'window_id': key,
                'window_start': window_start,
                'window_end': window_end,
                'snp_count': snp_count,
                'likelihood': likelihood
            })

''' all variants '''            
write_output('all_500kb.csv', chr1_norm_sfs, 500000, start_position=None, end_position=None, variant_type=None)

''' synonymous variants '''
write_output('syn_500kb.csv', chr1_norm_sfs_syn, 500000, start_position=None, end_position=None, variant_type='synonymous_variant')
    
''' nonsynonymous variants '''
write_output('nonsyn_500kb.csv', chr1_norm_sfs_nonsyn, 500000, start_position=None, end_position=None, variant_type='missense_variant')
    

''' end of applying pipeline on ECB data '''

##########

''' applying pipeline on simulations data '''

vcf_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs/concatenated_vcfs/gen.5000.concatenated.vcf.gz"
popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims.txt"

data_dict = make_data_dict_vcf(vcf_filename, popinfo_filename, snp_type=None)
background_sfs = calculate_2d_sfs(data_dict, 'p1', 'p2', start_position=0, end_position=500000)
normalized_background_sfs = normalize_2d_sfs(background_sfs)
p_values_sims = calculate_p_window(data_dict, normalized_background_sfs, 100000, 'p1', 'p2', start_position=None, end_position=None)



# calculate mean and std dev for background and background

background_p_values = []
foreground_p_values = []

for key, value in p_values_sims.items():
    window_start = int(key.split('_')[1].split('-')[0])  # Assuming window is formatted as 'start_end'
    if 0 <= window_start <= 1000000:
        background_p_values.append(value)
    elif 1000000 <= window_start <= 1499999:
        foreground_p_values.append(value)
        
background_p_values = np.array(background_p_values)
foreground_p_values = np.array(foreground_p_values)

mean_background = np.mean(background_p_values)
std_dev_background = np.std(background_p_values)

mean_foreground = np.mean(foreground_p_values)
std_dev_foreground = np.std(foreground_p_values)

print(f"background (0-1000000): mean = {mean_background}, std dev = {std_dev_background}")
print(f"foreground region (1000000-1499999): mean = {mean_foreground}, std dev = {std_dev_foreground}")

# store output (p_values_sims)

col_names = ['windows', 'p_values']

with open('loglikelihood_5000.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = col_names) 
    writer.writeheader() 
    
    for key, value in p_values_sims.items():
        writer.writerow({'windows': key, 'p_values': value}) 

'''
streamlining the likelihood inference across all files.
- input: path to main directory
- output: csv file with the following columns:
    column 1: generation # (from 5000 to 6000)
    column 2: chromosome name (foreground or background)
    column 3: window coordinates
    column 4: likelihood values
'''

def likelihood_scan(main_dir, output):
    
    # target_vcfs = []
    # concatenated_vcfs = []
    
    # define pop map file
    popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

    # get generation IDs
    generations = get_gens(main_dir)
    # print(patterns)

    
    col_names = ['generation', 'iteration', 'region', 'window_coords', 'likelihood']
    
    with open(output, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=col_names)
        writer.writeheader()
    
        # get list of target vcfs and concatenated vcf files
        for generation in generations:
            target_vcfs = glob.glob(f"{main_dir}/iter*/*{generation}*.vcf.gz")
            concatenated_vcfs = glob.glob(f"{main_dir}/concatenated_vcfs/gen.{generation}.concatenated.vcf.gz")
        
            # print(target_vcfs)
            # print(concatenated_vcfs)
                
                # get background sfs from concatenated vcfs
            for vcf in concatenated_vcfs:
                data_dict = make_data_dict_vcf(vcf, popinfo_filename, snp_type=None)
                background_sfs = calculate_2d_sfs(data_dict, 'p1', 'p2', start_position=0, end_position=500000)
                normalized_background_sfs = normalize_2d_sfs(background_sfs)
            
                # print(normalized_background_sfs)
    
                # get p values using target vcfs as input
                
                for vcf_input in target_vcfs:
                    data_dict_target = make_data_dict_vcf(vcf_input, popinfo_filename, snp_type=None)
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

main_dir = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs"

likelihood_scan(main_dir, 'likelihoods_500kb.csv')



# likelihood scan on concatenated files

popinfo_filename = "/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/popmap_sims_copy.txt"

# get generation IDs
generations = get_gens(main_dir)

col_names = ['generation', 'region', 'window_coords', 'likelihood']

with open('likelihoods_concatenated.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    writer.writeheader()
    
    for generation in generations:
        concatenated_vcfs = glob.glob(f"/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/simulations/results/vcfs/concatenated_vcfs/gen.{generation}.concatenated.vcf.gz")
        print(concatenated_vcfs)
        
        for vcf in concatenated_vcfs:
            data_dict = make_data_dict_vcf(vcf, popinfo_filename, snp_type=None)
            background_sfs = calculate_2d_sfs(data_dict, 'p1', 'p2', start_position=0, end_position=500000)
            normalized_background_sfs = normalize_2d_sfs(background_sfs)
            p_values_sims = calculate_p_window(data_dict, normalized_background_sfs, 100000, 'p1', 'p2', start_position=None, end_position=None)
    
            for key, value in p_values_sims.items():
                
                window_start, window_end = key.split('_')[1].split('-')
                region = 'background' if int(window_end) < 1000000 else 'foreground'
    
                writer.writerow({
                    'generation': generation,
                    'region': region,
                    'window_coords': key,
                    'likelihood': value
                })

''' end of trying pipeline on simulations data '''

##########

''' normalize 2D SFS extracted using dadi-cli '''

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

# chr1
sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chr1.downsampled.folded.fs'
norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chr1.downsampled.folded.normalized.fs'
normalize_dadi_sfs(sfs, norm_sfs)

# chrZ
sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chrZ_highestFSTwindow.downsampled.folded.fs'
norm_sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chrZ_highestFSTwindow.downsampled.folded.normalized.fs'
normalize_dadi_sfs(sfs, norm_sfs)


''' extract 1D-SFS data from dadi sf '''
    
def dadi_1D_sfs(sfs, output):
    
    with open(sfs, 'r') as f:
        lines = f.readlines()

    allele_counts = []
    for x in lines[1].strip().split():
        allele_counts.append(float(x))

    total_count = sum(allele_counts[1:-1])

    normalized_counts = []
    for count in allele_counts:
        normalized_count = count / total_count
        normalized_counts.append(normalized_count)

    # determine the maximum frequency
    max_freq = len(allele_counts) - 1

    # create a list of frequencies
    frequencies = list(range(max_freq + 1))

    # write the results to a CSV file
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['freq', 'count', 'normalized_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for freq, count, norm_count in zip(frequencies, allele_counts, normalized_counts):
            writer.writerow({'freq': freq, 'count': count, 'normalized_count': norm_count})

sfs = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chr1.bv.downsampled.folded.fs'
output = '/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/data_summer2024/chr1.bv.downsampled.folded.fs.csv'
dadi_1D_sfs(sfs, output)
