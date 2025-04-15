#ifndef WAVREADER_H
#define WAVREADER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

// Sample .wav header format
struct wav_header {
    uint32_t chunkID;
    uint32_t chunkSize;
    uint32_t format;
    uint32_t subChunk1ID;
    uint32_t subChunk1Size;
    uint16_t audioFormat;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;   
    uint16_t bits_per_sample;
    uint32_t subChunk2ID;
    uint32_t subChunk2Size;
};

// For reading files
struct FileStats {  
    std::map<std::vector<int16_t>, size_t> ngram_counts;
    size_t total_count = 0;
};

// TokenEntry in Dictionary
struct TokenEntry {
    size_t ID;
    size_t count = 0;
    double probability = 0.0;
    std::vector<int16_t> symbol_sequence;
};

// DICTIONARY TYPE
typedef std::vector<std::vector<int16_t>> Data;                                                 ///< QUANTISED RAW DATA
typedef std::vector<std::vector<TokenEntry>> TokenData;                                         ///< TOKENISED STREAM
typedef std::map<std::vector<int16_t>, TokenEntry> Dictionary;                                  ///< DICTIONARY OF TOKENS
typedef std::map<std::pair<std::vector<int16_t>, std::vector<int16_t>>, size_t> PairCount;      ///< COUNNT OF PAIRS

class wavReader {

    private:

        const int N;                ///< DICTIONARY SIZE 
        int num_files;              ///< NUMBER OF DATA FILES (.wav)
        size_t stream_size = 0;     ///< NUMBER OF DATAPOINTS IN DATASET

        FileStats globalStats;

        // QUANTISED DATA, TOKENISED DATA, PAIR COUNTS, STREAM OF TOKEN IDs
        Data quantData;
        TokenData tokenisedData;
        PairCount pairCounts;
        std::vector<std::vector<int>> idStreams;
        
        std::vector<std::pair<std::vector<int16_t>, double>> sortedProbs;
        
        Dictionary token_dict;

        size_t nextID = 0;
        size_t max_token_len = 1;

        std::vector<size_t> tokenCount;
        uint32_t totalSamples = 0;

        void readData(const std::string& path, const int idx);

        void makeSortedProb();
        
        void statsToDict();

        void initUnigrams(FileStats& stats, const int idx);

        void createTokenStream(const int idx);

        void countTokenPairs(PairCount& pair_counts, const int idx);

        void mergePair();      

        void countTokens();

        void exportTokenStream();                                                        

    public:
    
        void outputTxt();

        wavReader(int DICTIONARY_SIZE);

        ~wavReader();

        // GETTERS:
        const int getNumFiles();
        const int getNumTokens();
        const std::vector<int>& getTokenStream(const int idx); 
        size_t getStreamSize();
};

#endif // WAVREADER_H