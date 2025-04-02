#include "../include/wavReader.h"

//see https://github.com/phoboslab/neuralink_brainwire/blob/master/bwenc.c
//the code author implemented this method to downsample the data by 6 bits, as for some reason ...
//it's not done by just a simple bit shift.

static inline int brainwireQuant(int v) {
    return static_cast<int>(std::floor(v/64.0));
};

static inline int brainwireDequant(int v) {
    if (v >= 0) {
        return std::round(v * 64.061577 + 31.034184);
    }
    else {
        return -(std::round((-v - 1) * 64.061577 + 31.034184) - 1);
    }
};


void wavReader::makeSortedProb() {

    size_t divisor = this->globalStats.total_count;

    for (const auto &pair : this->globalStats.ngram_counts) {
        double probability = static_cast<double>(pair.second) / divisor;
        this->sortedProbs.emplace_back(pair.first, probability);
    }

    std::sort(this->sortedProbs.begin(), this->sortedProbs.end(),
         [](const std::pair<std::vector<int16_t>, double> &a, const std::pair<std::vector<int16_t>, double> &b)
         {
             return a.second > b.second;
        });

}

void wavReader::processFile(const std::string& path, FileStats& stats) {

    wav_header header;  

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    // see http://soundfile.sapp.org/doc/WaveFormat/
    
    file.read(reinterpret_cast<char*>(&header.chunkID),4);
    file.read(reinterpret_cast<char*>(&header.chunkSize),4);
    file.read(reinterpret_cast<char*>(&header.format),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1Size),4);
    file.read(reinterpret_cast<char*>(&header.audioFormat),2);
    file.read(reinterpret_cast<char*>(&header.num_channels),2);
    file.read(reinterpret_cast<char*>(&header.sample_rate),4);
    file.read(reinterpret_cast<char*>(&header.byte_rate),4);
    file.read(reinterpret_cast<char*>(&header.block_align),2);
    file.read(reinterpret_cast<char*>(&header.bits_per_sample),2);
    file.read(reinterpret_cast<char*>(&header.subChunk2ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk2Size),4);

    const size_t num_samples = header.subChunk2Size / (header.bits_per_sample / 8);

    int16_t* samples = new int16_t[num_samples];
    //int thread_num = omp_get_thread_num();

    //read audio data into samples
    file.read(reinterpret_cast<char*>(samples), header.subChunk2Size);

    if (file.gcount() != header.subChunk2Size) {
        std::cerr << "Error reading WAV data" << std::endl;
        file.close();
        delete[] samples;
        return;
    }

    // Quantise Data
    // for (size_t i = 0; i < num_samples; ++i) {
    //     samples[i] = brainwireQuant(samples[i]);
    // }

    // for (size_t i = 0; i + ngram_size <= num_samples; ++i) {
    //     std::vector<int16_t> ngram(ngram_size);
    //     for (int j = 0; j < ngram_size; ++j) {
    //         ngram[j] = samples[i + j];
    //     }
    //     stats.ngram_counts[ngram]++;
    //     stats.total_count++;
    // }

    file.close();

    // delete allocated memory
    delete[] samples;

    return;
}


void wavReader::readSingle(const std::string& path, FileStats& stats) {
    wav_header header;  

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    // see http://soundfile.sapp.org/doc/WaveFormat/
    // READ HEADER
    file.read(reinterpret_cast<char*>(&header.chunkID),4);
    file.read(reinterpret_cast<char*>(&header.chunkSize),4);
    file.read(reinterpret_cast<char*>(&header.format),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1Size),4);
    file.read(reinterpret_cast<char*>(&header.audioFormat),2);
    file.read(reinterpret_cast<char*>(&header.num_channels),2);
    file.read(reinterpret_cast<char*>(&header.sample_rate),4);
    file.read(reinterpret_cast<char*>(&header.byte_rate),4);
    file.read(reinterpret_cast<char*>(&header.block_align),2);
    file.read(reinterpret_cast<char*>(&header.bits_per_sample),2);
    file.read(reinterpret_cast<char*>(&header.subChunk2ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk2Size),4);

    const size_t num_samples = header.subChunk2Size / (header.bits_per_sample / 8);

    // ALLOCATE SAMPLES ON HEAP
    int16_t* samples = new int16_t[num_samples];

    //read audio data into samples
    file.read(reinterpret_cast<char*>(samples), header.subChunk2Size);

    if (file.gcount() != header.subChunk2Size) {
        std::cerr << "Error reading WAV data" << std::endl;
        file.close();
        delete[] samples;
        return;
    }

    // Quantise Data
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = brainwireQuant(samples[i]);
    }

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<int16_t> sample = {samples[i]};
        stats.ngram_counts[sample]++;
        stats.total_count++;
    }

    file.close();

    // delete allocated memory
    delete[] samples;

    return;
}

PairCount wavReader::countTokenPairs(const std::string& path) {
    std::map<std::pair<std::vector<int16_t>, std::vector<int16_t>>, size_t> pair_counts;

    wav_header header;  

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return pair_counts;
    }

    // see http://soundfile.sapp.org/doc/WaveFormat/
    // READ HEADER
    file.read(reinterpret_cast<char*>(&header.chunkID),4);
    file.read(reinterpret_cast<char*>(&header.chunkSize),4);
    file.read(reinterpret_cast<char*>(&header.format),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1Size),4);
    file.read(reinterpret_cast<char*>(&header.audioFormat),2);
    file.read(reinterpret_cast<char*>(&header.num_channels),2);
    file.read(reinterpret_cast<char*>(&header.sample_rate),4);
    file.read(reinterpret_cast<char*>(&header.byte_rate),4);
    file.read(reinterpret_cast<char*>(&header.block_align),2);
    file.read(reinterpret_cast<char*>(&header.bits_per_sample),2);
    file.read(reinterpret_cast<char*>(&header.subChunk2ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk2Size),4);

    const size_t num_samples = header.subChunk2Size / (header.bits_per_sample / 8);

    // ALLOCATE SAMPLES ON HEAP
    int16_t* samples = new int16_t[num_samples];

    //read audio data into samples
    file.read(reinterpret_cast<char*>(samples), header.subChunk2Size);

    if (file.gcount() != header.subChunk2Size) {
        std::cerr << "Error reading WAV data" << std::endl;
        file.close();
        delete[] samples;
        return pair_counts;
    }

    //Quantise
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = brainwireQuant(samples[i]);
    }

    std::vector<std::vector<int16_t>> tokens;
    size_t i = 0;
    while (i < num_samples) {
        bool match_found = false;
        const size_t max_len = std::min<size_t>(10, num_samples - i); 

        for (size_t len = max_len; len >= 1; --len) {
            std::vector<int16_t> candidate(samples + i, samples + i + len);

            if (token_dict.count(candidate)) {
                tokens.push_back(candidate);
                i += len;
                match_found = true;
                break;
            }
        }

        if (!match_found) {
            std::vector<int16_t> fallback = {samples[i]};
            tokens.push_back(fallback);
            i++;
        }
    }

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        std::pair<std::vector<int16_t>, std::vector<int16_t>> pair = {tokens[i], tokens[i+1]};
        pair_counts[pair]++;
    }

    delete[] samples;
    file.close();
    return pair_counts;   

}

void wavReader::mergePair() {
    if (this->global_pair_counts.empty()) {
        std::cerr << "ERROR : NO PAIRS TO MERGE" << std::endl;
        return;
    }

    auto best_pair_it = std::max_element(global_pair_counts.begin(), global_pair_counts.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        }
    );

    const std::pair<std::vector<int16_t>, std::vector<int16_t>>& best_pair = best_pair_it->first;
    size_t best_count = best_pair_it->second;

    std::cout << "MERGING PAIR : [";
    for (int16_t v : best_pair.first) std::cout << v << " ";
    std::cout << "] + [";
    for (int16_t v : best_pair.second) std::cout << v << " ";
    std::cout << "] (count = " << best_count << ")" << std::endl;

    std::vector<int16_t> merged;
    merged.insert(merged.end(), best_pair.first.begin(), best_pair.first.end());
    merged.insert(merged.end(), best_pair.second.begin(), best_pair.second.end());

    //ADD TO DICTIONARY
    if (this->token_dict.count(merged) == 0) {
        TokenEntry entry;
        entry.symbol_sequence = merged;
        entry.ID = this->nextID++;
        entry.count = best_count;
        entry.probability = 0.0;

        this->token_dict[merged] = entry;
    }

    std::cout << "CURRENT DICTIONARY SIZE : " << this->token_dict.size() << std::endl;

    this->global_pair_counts.clear();
}


void wavReader::outputTxt() {

    double prob_sum = 0.0;

    // Write output to .txt
    std::ofstream output_file("ngram_probabilities.txt");
    
    if (!output_file.is_open())
    {
        std::cerr << "Cannot open output file ..." << std::endl;
        return;
    }
    
    output_file << "Symbol Probabilities:\n";
    //std::cout << "\nSymbol Probabilities:\n";
    for (const auto& [first, second] : this->sortedProbs)
    {   
        prob_sum += second;
        output_file << "Symbol: ["; 
        //std::cout << "Symbol: [";
        for (size_t i = 0; i < first.size(); ++i) {
            output_file << first[i];
            //std::cout << first[i];
            if (i != first.size() - 1) {
                output_file << ", ";
                //std::cout << ", ";
            }
        }
        output_file << "] | Probability: " << std::fixed << std::setprecision(20) << second << "\n";
        //std::cout << "] | Probability: " << std::fixed << std::setprecision(20) << second << "\n";
    }

    std::cout << "\nSum of all probabilities: " << prob_sum <<  " , num symbols = " << this->sortedProbs.size() << std::endl;
    
    return;
}

void wavReader::statsToDict() {

    for (auto& [ngram, entry] : token_dict) {
        if (globalStats.ngram_counts.count(ngram)) {
            entry.count = globalStats.ngram_counts[ngram];
            entry.probability = static_cast<double>(entry.count) / globalStats.total_count;
        } else {
            entry.count = 0;
            entry.probability = 0.0;
        }
    }

    for (const auto& [ngram, count] : globalStatsFinal.ngram_counts) {
        if (token_dict.count(ngram) == 0) {
            TokenEntry entry;
            entry.symbol_sequence = ngram;
            entry.ID = nextID++;
            entry.count = count;
            entry.probability = static_cast<double>(count) / globalStatsFinal.total_count;

            token_dict[ngram] = entry;
        }
    }

    // Now sort by probability
    sortedProbs.clear();
    for (const auto& [sym, entry] : token_dict) {
        sortedProbs.emplace_back(sym, entry.probability);
    }

    std::sort(sortedProbs.begin(), sortedProbs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    std::cout << "CURRENT DICTIONARY SIZE : " << token_dict.size() << std::endl;
}

void wavReader::readMany(const std::string& path, FileStats& stats) {
    wav_header header;  

    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    // see http://soundfile.sapp.org/doc/WaveFormat/
    // READ HEADER
    file.read(reinterpret_cast<char*>(&header.chunkID),4);
    file.read(reinterpret_cast<char*>(&header.chunkSize),4);
    file.read(reinterpret_cast<char*>(&header.format),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk1Size),4);
    file.read(reinterpret_cast<char*>(&header.audioFormat),2);
    file.read(reinterpret_cast<char*>(&header.num_channels),2);
    file.read(reinterpret_cast<char*>(&header.sample_rate),4);
    file.read(reinterpret_cast<char*>(&header.byte_rate),4);
    file.read(reinterpret_cast<char*>(&header.block_align),2);
    file.read(reinterpret_cast<char*>(&header.bits_per_sample),2);
    file.read(reinterpret_cast<char*>(&header.subChunk2ID),4);
    file.read(reinterpret_cast<char*>(&header.subChunk2Size),4);

    const size_t num_samples = header.subChunk2Size / (header.bits_per_sample / 8);

    // ALLOCATE SAMPLES ON HEAP
    int16_t* samples = new int16_t[num_samples];

    //read audio data into samples
    file.read(reinterpret_cast<char*>(samples), header.subChunk2Size);

    if (file.gcount() != header.subChunk2Size) {
        std::cerr << "Error reading WAV data" << std::endl;
        file.close();
        delete[] samples;
        return;
    }

    // Quantise Data
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = brainwireQuant(samples[i]);
    }

    size_t i = 0;
    while (i < num_samples) {
        size_t max_len = std::min<size_t>(10, num_samples - i);
        bool match = false;

        for (size_t len = max_len; len >= 1; --len) {
            std::vector<int16_t> sub(samples + i, samples + i + len);
            if (this->token_dict.count(sub)) {
                stats.ngram_counts[sub]++;
                stats.total_count++;
                i += len;
                match = true;
                break;
            }
        }

        if (!match) {
            //FALLBACK
            std::cout << "ERROR NOT FOUND IN DICTIONARY" << std::endl;
            std::vector<int16_t> fallback = {samples[i]};
            stats.ngram_counts[fallback]++;
            stats.total_count++;
            i++;
        }
    }


    file.close();

    // delete allocated memory
    delete[] samples;

    return;
}



wavReader::wavReader(int DICTIONARY_SIZE) : N(DICTIONARY_SIZE) {

    const std::string directory = "./data/";    
    std::vector<std::string> file_paths;

    // POPULATE file_paths
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
		if (entry.is_regular_file()) {

			std::string path = entry.path().string();
            file_paths.push_back(path);
        }
    }
    
    int num_files = file_paths.size();

    {
    std::vector<FileStats> allStats(num_files);

    std::cout << "READING " << num_files << " FILES ..." << std::endl;

    // FIRST CREATE SINGLE SYMBOLS IN DICTIONARY:
    #pragma omp parallel for 
    for (int i = 0; i < num_files;  ++i) {
        wavReader::readSingle(file_paths[i], allStats[i]);
    }

    // MERGE THREADS INTO SINGLE
    for (const FileStats& local : allStats) {
        for (const auto& [ngram, count] : local.ngram_counts) {
            this->globalStats.ngram_counts[ngram] += count;
        }
        this->globalStats.total_count += local.total_count;
    }
    }

    // MERGE COUNTS TO DICTIONARY:
    wavReader::statsToDict();

    // NOW DO BPE:
    while (this->token_dict.size() < this->N) {

        global_pair_counts.clear();

        // 1 - FIND MOST COMMON PAIR
        std::vector<PairCount> local_counts(num_files);

        // 2 - RUN THROUGH DATASET IN PARALLEL
        #pragma omp parallel for
        for (int i = 0; i < num_files; ++i) {
            local_counts[i] = wavReader::countTokenPairs(file_paths[i]);
        }

        // 3 - ADD UP CONTRIBUTIONS
        for (const PairCount& local : local_counts) {
            for (const auto& [pair, count] : local) {
                this->global_pair_counts[pair] += count;
            }
        }

        // 4 - MERGE MOST COMMON PAIR TO ONE SYMBOL
        wavReader::mergePair();
    }

    // RUN THROUGH DATASET ONE LAST TIME:
    std::vector<FileStats> allStats(num_files);

    #pragma omp parallel for 
    for (int i = 0; i < num_files;  ++i) {
        wavReader::readMany(file_paths[i], allStats[i]);
    }

    for (const FileStats& local : allStats) {
        for (const auto& [ngram, count] : local.ngram_counts) {
            this->globalStatsFinal.ngram_counts[ngram] += count;
        }
        this->globalStatsFinal.total_count += local.total_count;
    }

    this->token_dict.clear();
    wavReader::statsToDict();

    wavReader::outputTxt();
}

wavReader::~wavReader() {

    std::cout << "READING TERMINATED" << std::endl;

}