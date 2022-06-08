#pragma once

#include <string>

using std::string;

template <typename K, typename V>
class HashNode {
    public:
        HashNode() = default;
        HashNode(const K& key, const V& value) {
            key_ = key;
            value_ = value;
        }

        string ToString() {
            return "Key: " + std::to_string(key) + " | Value: " + std::to_string(value);
        }

        K key_;
        V value_;

        HashNode* next_ = nullptr;
};