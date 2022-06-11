#pragma once

#include "cuda_runtime.h"

#include "cuda_handler.hpp"

#include <stdexcept>
#include <string>
#include <iostream>
#include <functional>
#include <vector>

#include <thrust/universal_vector.h>

using std::string;
using std::function;
using std::vector;
using std::reference_wrapper;

template <typename K>
struct HashFunc {
    __host__ __device__ size_t operator()(const K& key, size_t size) const
    {
        size_t hash = (size_t) (key);
        return (hash << 1) % size;
    }
};

template <typename K, typename V, typename F = HashFunc<K>>
class HashMap {
    public:
        //Constructor
        HashMap() {
            hash_table_size_ = DEFAULT_SIZE;
            Initialization();
        }

        HashMap(const size_t& hash_table_size) {
            if (hash_table_size < 1) {
                throw std::invalid_argument("The input size for the hash table should be at least 1!");
            }
            hash_table_size_ = hash_table_size;
            Initialization();
        }

        void Initialization() {
            cudaMallocManaged(&table_, (size_t)sizeof(V) * hash_table_size_);
            cudaMallocManaged(&hashes_, (size_t) sizeof(long) * hash_table_size_);
        }

        void Synchronize() {
            cudaError_t cuda_status = cudaSuccess;

            function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
            cuda_status = WrapperFunction(sync_func, "HashMap", "HashMap", cuda_status, "");
        }

        //Destructor
        ~HashMap() {
            //Synchronize();
            cudaFree(table_);
            cudaFree(hashes_);
        }

        //Memory Allocation/Deallocation Unified Overloads
        void* operator new(size_t size) {
            void* ptr;
            cudaMallocManaged(&ptr, sizeof(HashMap<K, V, HashFunc<K>>)); //Allocates the size of the 
            cudaDeviceSynchronize();
            return ptr;
        }

        void operator delete(void* ptr) {
            cudaDeviceSynchronize();
            cudaFree(ptr);
        }

        //Associative Array Logic
        __host__ __device__ long FindHash(const long& hash) {
            if (hash > hash_table_size_ || hashes_[hash] == 0) {
                return -1;
            }
            return hashes_[hash] - 1;
        }

        //General Data Accessor Methods

        __host__ __device__ V Get(const K& key) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                printf("Invalid Index!");
            }
            return table_[hash_pos];
            
        }

        __host__ void Put(const K& key, const V& value) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                hashes_[hash] = size_ + 1;

                table_[size_] = value;
                size_++;
            }
            else {
                table_[hash_pos] = value;
            }
        }

        __host__ __device__ void Remove(const K& key) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                return;
            }

            table_[hash_pos] = 0;
            hashes_[hash] = -1;

            Synchronize();

            size_--;
        }

        //Output Logic
        string ToString() {
            string output;
            for (const auto* node : table_) {
                output += node->ToString();
            }
            return output;
        }

        //Accessor Overloads

        __host__ __device__ V operator[](const K& key) {
            return Get(key);
        }

        __host__ __device__ V& operator[](const int& index) {
            return table_[index]->value_;
        }

        //Assignment Operator Overload (Shallow Copy)
        HashMap& operator=(const HashMap& src) {
            if (src == *this) {
                return *this;
            }
            table_ = src.table_;
            return *this;
        }

        long size_ = 0;
        size_t hash_table_size_;

    private:
        V* table_;
        long* hashes_;

        F hash_func_;

        const size_t DEFAULT_SIZE = 10000;
};
