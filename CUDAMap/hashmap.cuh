#pragma once

#include "cuda_runtime.h"

#include "cuda_handler.hpp"
#include "hashnode.cuh"

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
        size_t hash = std::hash<size_t>()(key);
        return (hash << 1) % size;
    }
};

template <typename K, typename V, typename F = HashFunc<K>>
class HashMap {
    public:
        //Constructor
        HashMap() {
            hash_table_size_ = DEFAULT_SIZE;
            hashes_ = thrust::universal_vector<long>(hash_table_size_, -1);
        }

        HashMap(const size_t& hash_table_size) {
            if (hash_table_size < 1) {
                throw std::invalid_argument("The input size for the hash table should be at least 1!");
            }
            hash_table_size_ = hash_table_size;
            hashes_ = thrust::universal_vector<long>(hash_table_size_, -1);
        }

        //Destructor
        ~HashMap() {
            cudaError_t cuda_status = cudaSuccess;
            CudaMemoryFreer<HashNode<K, V>>(allocs_);
            function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
            cuda_status = WrapperFunction(sync_func, "DestroyManagedMemory", "~HashMap", cuda_status, "");
        }

        //Associative Array Logic
        __host__ __device__ long FindHash(const long& hash) {
            if (hash > hash_table_size_) {
                return -1;
            }
            return hashes_[hash];
        }

        //General Data Accessor Methods

        __host__ __device__ void Get(const K& key, V& value) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                return;
            }
            value = table_[hash_pos]->value_;
        }

        __host__ __device__ void Put(const K& key, const V& value) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                hashes_[hash] = size_;

                HashNode<K, V>* node = nullptr;
                cudaMallocManaged(&node, (size_t) sizeof(HashNode<K,V>));

                function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
                WrapperFunction(sync_func, "MallocManagedMemory", "Put", cudaSuccess, "");

                table_.push_back(node);
                allocs_.push_back(node);

                node->key_ = key;
                node->value_ = value;
            }
            else {
                table_[hash_pos]->value_ = value;
            }
            size_++;
        }

        __host__ __device__ void Remove(const K& key) {
            size_t hash = hash_func_(key, hash_table_size_);
            long hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                return;
            }
            table_.erase(table_.begin, hash_pos);
            hashes_.erase(hashes_.begin() + hash_pos);

            cudaFree(allocs_[hash_pos].get());
            allocs_.erase(allocs_.begin() + hash_pos);

            function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
            WrapperFunction(sync_func, "DestroyManagedMemory", "Remove", cuda_status, "");

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

        __host__ __device__ V& operator[](const K& key) {
            V value;
            Get(key, value);
            return value;
        }

        __host__ __device__ V& operator[](const int& index) {
            V value;
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
        size_t hash_table_size_ = 0;

    private:
        thrust::universal_vector<HashNode<K, V>*> table_;
        thrust::universal_vector<long> hashes_;

        vector<reference_wrapper<HashNode<K, V>*>> allocs_;

        F hash_func_;

        const size_t DEFAULT_SIZE = 10000;
};