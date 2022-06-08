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
    size_t operator()(const K& key) const
    {
        size_t hash = std::hash<size_t>()(key);
        return hash << 1;
    }
};

template <typename K, typename V, typename F = HashFunc<K>>
class HashMap {
    public:
        //Constructor
        HashMap() = default;

        //Destructor
        ~HashMap() {
            cudaError_t cuda_status = cudaSuccess;
            CudaMemoryFreer<HashNode<K, V>>(allocs_);
            function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
            cuda_status = WrapperFunction(sync_func, "DestroyManagedMemory", "~HashMap", cuda_status, "");
        }

        //Associative Array Logic
        size_t FindHash(const size_t& hash) {
            for (size_t i = 0; i < size_; i++) {
                if ((unsigned long) hashes_[i] == (unsigned long) hash) {
                    return i;
                }
            }
            return -1;
        }

        //General Data Accessor Methods

        void Get(const K& key, V& value) {
            size_t hash = hash_func_(key);
            size_t hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                return;
            }
            value = table_[hash_pos]->value_;
        }

        void Put(const K& key, const V& value) {
            size_t hash = hash_func_(key);
            size_t hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                hashes_.push_back(hash);

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

        void Remove(const K& key) {
            size_t hash = hash_func_(key);
            size_t hash_pos = FindHash(hash);
            if (hash_pos == -1) {
                return;
            }
            table_.erase(hash_pos);
            hashes_.erase(hash_pos);

            cudaFree(allocs_[hash_pos].get());
            allocs_.erase(hash_pos);

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

        //Accessor Overload
        V& operator[](const K& key) {
            V value;
            Get(key, value);
            return value;
        }

        //Assignment Operator Overload (Shallow Copy)
        HashMap& operator=(const HashMap& src) {
            if (src == *this) {
                return *this;
            }
            table_ = src.table_;
            return *this;
        }

        unsigned int size_ = 0;

    private:
        thrust::universal_vector<HashNode<K, V>*> table_;
        thrust::universal_vector<size_t> hashes_;

        vector<reference_wrapper<HashNode<K, V>*>> allocs_;

        F hash_func_;
};