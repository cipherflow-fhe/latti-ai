#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "interface/inference_server.h"

namespace inference_detail {
Bytes read_eval_context_file(std::istream& file, const std::string& filepath);
}

class ReportedSizeBuffer : public std::stringbuf {
public:
    explicit ReportedSizeBuffer(std::streamoff size) : std::stringbuf("short"), size_(size) {}

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode mode) override {
        if (offset == 0 && direction == std::ios_base::cur) {
            return pos_type(size_);
        }
        return std::stringbuf::seekoff(offset, direction, mode);
    }

private:
    std::streamoff size_;
};

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Expected task directory");
        }
        InferenceServer server(std::string(argv[1]) + "/server");
        bool rejected = false;
        try {
            server.import_eval_context_file(std::string(argv[1]) + "/missing-evaluation-context");
        } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) {
            throw std::runtime_error("Missing file accepted");
        }
        for (auto size :
             {std::streamoff(-1), std::streamoff(0), std::streamoff(100), std::numeric_limits<std::streamoff>::max()}) {
            ReportedSizeBuffer buffer(size);
            std::istream file(&buffer);
            rejected = false;
            try {
                inference_detail::read_eval_context_file(file, "test stream");
            } catch (const std::exception&) { rejected = true; }
            if (!rejected) {
                throw std::runtime_error("Invalid length or short read accepted");
            }
        }
        std::istringstream valid("payload", std::ios::binary | std::ios::ate);
        valid.seekg(0, std::ios::end);
        auto data = inference_detail::read_eval_context_file(valid, "valid stream");
        if (std::string(data.begin(), data.end()) != "payload") {
            throw std::runtime_error("Valid payload read incorrectly");
        }
        auto parameters = fhe_ops_lib::CkksParameter::create_parameter(4096);
        auto path = std::filesystem::temp_directory_path() /
                    ("latti-eval-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::filesystem::remove(path);
            }
        } cleanup{path};
        { std::ofstream empty(path); }
        rejected = false;
        try {
            server.import_eval_context_file(path.string());
        } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) {
            throw std::runtime_error("Empty file accepted");
        }
        auto client = fhe_ops_lib::CkksContext::create_streaming_context(parameters, path.string());
        server.import_eval_context_file(path.string());

        // Exercise the default client entry point and file export after persistence/release.
        auto client_directory = path.string() + "-client";
        struct DirectoryCleanup {
            std::filesystem::path path;
            ~DirectoryCleanup() {
                std::filesystem::remove_all(path);
            }
        } client_cleanup{client_directory};
        std::filesystem::create_directory(client_directory);
        for (const auto* name : {"task_config.json", "ckks_parameter.json"}) {
            std::filesystem::copy_file(std::filesystem::path(argv[1]) / "client" / name,
                                       std::filesystem::path(client_directory) / name);
        }
        InferenceClient default_client(client_directory);
        default_client.setup();
        auto published = std::filesystem::path(client_directory) / "eval_context.bin";
        auto size = std::filesystem::file_size(published);
        if (size == 0) {
            throw std::runtime_error("setup returned before file publication");
        }
        auto saved_client = default_client.export_full_context();
        default_client.release();
        rejected = false;
        try {
            default_client.export_eval_context_to_file(path.string());
        } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) {
            throw std::runtime_error("File export accepted a released client");
        }
        default_client.load_full_context(saved_client);
        default_client.export_eval_context_to_file(path.string());
        if (std::filesystem::file_size(path) != size) {
            throw std::runtime_error("Restored client exported an incomplete evaluation context");
        }
        server.import_eval_context_file(path.string());
        std::cout << "Evaluation context file read and failure tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
