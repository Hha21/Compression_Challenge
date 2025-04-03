#include "../include/wavReader.h"

//see https://github.com/phoboslab/neuralink_brainwire/blob/master/bwenc.c
//the code author implemented this method to downsample the data by 6 bits, as for some reason ...
//it's not done by just a simple bit shift.

static inline int16_t brainwireQuant(int16_t v) {
    return static_cast<int16_t>(std::floor(v/64.0));
};

static inline int16_t brainwireDequant(int16_t v) {
    if (v >= 0) {
        return std::round<int16_t>(v * 64.061577 + 31.034184);
    }
    else {
        return -(std::round<int16_t>((-v - 1) * 64.061577 + 31.034184) - 1);
    }
};

void wavReader::makeSortedProb() {
    sortedProbs.clear();
    for (const auto& [sym, entry] : token_dict) {
        sortedProbs.emplace_back(sym, entry.probability);
    }

    std::sort(sortedProbs.begin(), sortedProbs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
}

void wavReader::readData(const std::string& path, const int idx) {
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

    std::vector<int16_t> newData;
    newData.resize(num_samples);

    //read audio data into samples
    file.read(reinterpret_cast<char*>(samples), header.subChunk2Size);

    if (file.gcount() != header.subChunk2Size) {
        std::cerr << "Error reading WAV data" << std::endl;
        file.close();
        delete[] samples;
        return;
    }


    for (size_t i = 0; i < num_samples; ++i) {
        newData[i] = brainwireQuant(samples[i]);
    }

    this->quantData[idx] = newData;

    file.close();

    // delete allocated memory
    delete[] samples;

    return;
}

void wavReader::initUnigrams(FileStats& stats, const int idx) {
    std::vector<int16_t>& localData = this->quantData[idx];

    size_t num_samples = localData.size();

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<int16_t> sample = {localData[i]};
        stats.ngram_counts[sample]++;
        stats.total_count++;
    }
}

void wavReader::createTokenStream(const int idx) {
    const std::vector<int16_t>& localData = this->quantData[idx];
    size_t& localCount = this->tokenCount[idx];
    localCount = 0;

    size_t num_samples = localData.size();

    std::vector<TokenEntry> stream;

    size_t i = 0;

    while (i < num_samples) {
        bool match = false;

        // FIND LONGEST MATCHING TOKEN
        size_t max_len = std::min(this->max_token_len, num_samples - i);

        for (size_t len = max_len; len >= 1; --len) {
            std::vector<int16_t> candidate(localData.begin() + i, localData.begin() + i + len);

            auto it = this->token_dict.find(candidate);
            if (it != token_dict.end()) {
                stream.push_back(it->second);
                localCount++;
                i += len;
                match = true;
                break;
            }
        }

        if (!match) {
            std::vector<int16_t> err = {localData[i]};
            std::cout << "ERROR TOKEN " << localData[i] << " | NOT FOUND IN DICTIONARY!" << std::endl;
            ++i;
        }
    }

    this->tokenisedData[idx] = std::move(stream);
}

void wavReader::countTokenPairs(PairCount& pair_counts, const int idx) {
  
    const std::vector<TokenEntry>& stream = this->tokenisedData[idx];
    size_t stream_len = stream.size();

    for (size_t i = 0; i + 1 < stream_len; ++i) {
        const std::vector<int16_t>& first =  stream[i].symbol_sequence;
        const std::vector<int16_t>& second = stream[i + 1].symbol_sequence;

        pair_counts[{first, second}]++;
    }
}

void wavReader::mergePair() {
    if (this->pairCounts.empty()) {
        std::cerr << "ERROR : NO PAIRS TO MERGE" << std::endl;
        return;
    }

    auto best_pair_it = std::max_element(pairCounts.begin(), pairCounts.end(),
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

    if (merged.size() > this->max_token_len) {
        this->max_token_len = merged.size();
    }

    //ADD TO DICTIONARY
    if (this->token_dict.count(merged) == 0) {
        TokenEntry entry;
        entry.symbol_sequence = merged;
        entry.ID = this->nextID++;
        entry.count = 0;
        entry.probability = 0.0;

        this->token_dict[merged] = entry;
    }

    std::cout << "CURRENT DICTIONARY SIZE : " << this->token_dict.size() << std::endl;

    this->pairCounts.clear();
}

void wavReader::countTokens() {
    size_t total_tokens = 0;

    for (const std::vector<TokenEntry> stream : this->tokenisedData) {
        for (const TokenEntry& token : stream) {
            this->token_dict[token.symbol_sequence].count++;
            total_tokens++;
        }
    }

    for (auto& [sym, entry] : this->token_dict) {
        entry.probability = static_cast<double>(entry.count) / total_tokens;
    }
}

void wavReader::outputTxt() {

    double prob_sum = 0.0;

    // Write output to .txt
    std::ofstream output_file("BPE_probabilities.txt");
    
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

    this->token_dict.clear();

    for (const auto& [ngram, count] : globalStats.ngram_counts) {
        if (token_dict.count(ngram) == 0) {
            TokenEntry entry;
            entry.symbol_sequence = ngram;
            entry.ID = this->nextID++;
            entry.count = 0;
            entry.probability = 0.0;

            token_dict[ngram] = entry;
        }
    }

    std::cout << "CURRENT DICTIONARY SIZE : " << token_dict.size() << std::endl;
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
    this->quantData.resize(num_files);
    this->tokenisedData.resize(num_files);
    this->tokenCount.resize(num_files);

    std::cout << "READING " << num_files << " FILES ..." << std::endl;

    // READ IN QUANTISED DATA ONCE:
    #pragma omp parallel for 
    for (int i = 0; i < num_files;  ++i) {
        wavReader::readData(file_paths[i], i);
    }

    {
    std::vector<FileStats> allStats(num_files);

    // FIND UNIGRAMS
    #pragma omp parallel for 
    for (int i = 0; i < num_files; ++i) {
        wavReader::initUnigrams(allStats[i], i);
    }

    // COLLAPSE PARALLEL STREAMS
    for (const FileStats& local : allStats) {
        for (const auto& [ngram, count] : local.ngram_counts) {
            this->globalStats.ngram_counts[ngram] += count;
        }
        this->globalStats.total_count += local.total_count;
    }
    }

    // WRITE DATA TO DICTIONARY
    wavReader::statsToDict();

    while (this->token_dict.size() < this->N) {

        // 1 - CREATE GREEDY TOKEN STREAM WITH CURRENT DICTIONARY
        #pragma omp parallel for 
        for (int i = 0; i < num_files; ++i) {
            wavReader::createTokenStream(i);
        }

        size_t stream_size = 0;
        for (int i = 0; i < num_files; ++i) {
            stream_size += this->tokenCount[i];
        }
        std::cout << "CURRENT NUMBER OF TOKENS IN STREAM: " << stream_size << std::endl;

        // COUNT PAIRS
        std::vector<PairCount> allCounts(num_files);

        #pragma omp parallel for 
        for (int i = 0; i < num_files; ++i) {
            wavReader::countTokenPairs(allCounts[i], i);
        }

        // COLLAPSE PARALLEL STREAMS
        for (const PairCount& local : allCounts) {
            for (const auto& [pair, count] : local) {
                this->pairCounts[pair] += count;
            }
        }

        // MERGE MOST FREQUENT PAIR
        wavReader::mergePair();
    }

    // CREATE ONE FINAL STREAM WITH FINAL DICTIONARY
    #pragma omp parallel for 
    for (int i = 0; i < num_files; ++i) {
        wavReader::createTokenStream(i);
    }

    // COUNT TOKENS IN TOKEN STREAM -> TOKEN_DICT
    wavReader::countTokens();

    // MAKE PROB AND OUTPUT TO TXT
    wavReader::makeSortedProb();
    wavReader::outputTxt();
}

wavReader::~wavReader() {
    std::cout << "READING TERMINATED" << std::endl;
}