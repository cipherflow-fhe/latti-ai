#include "../data_structs/feature0d.h"
#include "mpc_task_meta_data.h"
#include <cstdarg>

using namespace std;

Bytes MpcTaskMetaData::serialize() const {
    stringstream ss;
    size_t n_proto = types.size();
    ss_write(ss, n_proto);
    for (int i = 0; i < n_proto; i++) {
        ss_write(ss, types[i]);
        ss_write_vector(ss, data[i]);
    }
    Bytes bytes = ss_to_bytes(ss);
    return bytes;
}

void MpcTaskMetaData::deserialize(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    size_t n_proto;
    ss_read(ss, &n_proto);
    types.resize(n_proto);
    data.resize(n_proto);
    for (int i = 0; i < n_proto; i++) {
        ss_read(ss, &types[i]);
        ss_read_vector(ss, &data[i]);
    }
}

Bytes _va_to_bytes(const vector<string>& fmt, va_list& args) {
    stringstream ss;
    for (const string& f : fmt) {
        if (f == "u8") {
            uint8_t x = va_arg(args, int);
            ss_write(ss, x);
        } else if (f == "u32") {
            uint32_t x = va_arg(args, uint32_t);
            ss_write(ss, x);
        } else if (f == "duo") {
            Duo x = va_arg(args, Duo);
            ss_write(ss, x);
        }
    }
    Bytes bytes = ss_to_bytes(ss);    
    return bytes;
}

Bytes va_to_bytes(vector<string> fmt, ...) {
    va_list args;
    va_start(args, fmt);
    Bytes bytes = _va_to_bytes(fmt, args);
    va_end(args);
    return bytes;
}

void bytes_to_va(const Bytes& bytes, vector<string> fmt, ...) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    va_list args;
    va_start(args, fmt);
    for (const string& f : fmt) {
        if (f == "u8") {
            uint8_t* x = va_arg(args, uint8_t*);
            ss_read(ss, x);
        } else if (f == "u32") {
            uint32_t* x = va_arg(args, uint32_t*);
            ss_read(ss, x);
        } else if (f == "duo") {
            Duo* x = va_arg(args, Duo*);
            ss_read(ss, x);
        }
    }
    va_end(args);
}

void MpcTaskMetaData::append(MpcProtoType type, vector<string> fmt, ...) {
    va_list args;
    va_start(args, fmt);
    Bytes bytes = _va_to_bytes(fmt, args);
    va_end(args);

    types.push_back(type);
    data.push_back(bytes);
}
