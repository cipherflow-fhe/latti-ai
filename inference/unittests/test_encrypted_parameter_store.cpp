// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <chrono>
#include <fstream>
#include "interface/encrypted_parameter_store.h"
#include "interface/runner_bundle_io.h"
#include "util/serial.h"

namespace fs = std::filesystem;
using namespace fhe_ops_lib;

TEST_CASE("lazy store evicts safely and supports containers without an index", "[parameter_store]") {
    const auto param = CkksParameter::create_parameter(8192);
    auto ctx = CkksContext::create_random_context(param);
    const int level = 1;
    const double scale = param.get_default_scale() * 0.625;
    std::vector<Bytes> payloads;
    for (double value : {0.25, -0.5, 0.75}) {
        payloads.push_back(ctx.encrypt_asymmetric(ctx.encode(std::vector<double>(param.get_n() / 2, value),
                                                            level, scale)).serialize(param));
    }
    const auto root = fs::temp_directory_path() / ("latti-store-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{root};
    std::stringstream container;
    ss_write(container, uint64_t{3});
    json spans = json::array();
    for (const auto& payload : payloads) {
        spans.push_back({{"offset", static_cast<uint64_t>(container.tellp()) + sizeof(size_t)},
                         {"length", payload.size()}});
        ss_write_vector(container, payload);
    }
    write_binary_file(root / "w.bin", ss_to_bytes(container));
    json argument = {{"id", "w"}, {"file", "w.bin"}, {"level", level},
                     {"scale", scale}, {"size", {3}}, {"layer_id", "linear"},
                     {"layout", {{"kind", "pcm_row_chunk_v1"}}}};
    for (bool indexed : {false, true}) {
        INFO("indexed=" << indexed);
        if (indexed) argument["elements"] = spans;
        json manifest = {{"arguments", {{"w", argument}}}};
        std::ofstream(root / "manifest.json") << manifest;
        EncryptedParameterStore store(root, payloads[0].size());
        auto first = store.load_element("w", 0);
        REQUIRE(store.load_element("w", 0) == first);
        auto second = store.load_element("w", 1);
        auto stats = store.cache_statistics();
        REQUIRE(stats["canonical_loads"] == 2);
        REQUIRE(stats["seed_loads"] == 2);
        REQUIRE(stats["layers"]["linear"]["loads"] == 2);
        REQUIRE(stats["cache_hits"] == 1);
        REQUIRE(stats["cache_evictions"] == 1);
        REQUIRE(stats["cache_peak_bytes"].get<uint64_t>() <= payloads[0].size());
        // Eviction releases only the store's reference; in-flight graph branches
        // continue to own the same valid, unmodified ciphertext.
        REQUIRE(ctx.decode(ctx.decrypt(*first))[0] == Approx(0.25).margin(1e-5));
        auto reloaded = store.load_element("w", 0);
        REQUIRE(reloaded != first);
        REQUIRE(reloaded->get_scale() == Approx(scale));
        REQUIRE(ctx.decode(ctx.decrypt(*second))[0] == Approx(-0.5).margin(1e-5));
        REQUIRE(ctx.decode(ctx.decrypt(*reloaded))[0] == Approx(0.25).margin(1e-5));
        REQUIRE_THROWS_WITH(store.load_element("w", 3), Catch::Contains("out of range"));
        REQUIRE_THROWS_WITH(store.load_element("missing", 0), Catch::Contains("not found"));

        EncryptedParameterStore uncached(root, 0);
        REQUIRE(uncached.load_element("w", 2) != uncached.load_element("w", 2));
        REQUIRE(uncached.cache_statistics()["cache_bytes"] == 0);
        REQUIRE(uncached.cache_statistics()["canonical_loads"] == 2);
        REQUIRE(uncached.cache_statistics()["seed_loads"] == 2);

        EncryptedParameterStore eager(root);
        auto& all = eager.load_argument("w");
        REQUIRE(all.size() == 3);
        REQUIRE(ctx.decode(ctx.decrypt(all[2]))[0] == Approx(0.75).margin(1e-5));

        manifest["arguments"]["w"]["scale"] = scale * 2;
        std::ofstream(root / "manifest.json") << manifest;
        EncryptedParameterStore wrong_scale(root);
        REQUIRE_THROWS_WITH(wrong_scale.load_element("w", 0), Catch::Contains("scale mismatch"));
        REQUIRE_THROWS_WITH(wrong_scale.load_argument("w"), Catch::Contains("scale mismatch"));
        manifest["arguments"]["w"]["scale"] = scale;
        manifest["arguments"]["w"]["level"] = level + 1;
        std::ofstream(root / "manifest.json") << manifest;
        EncryptedParameterStore wrong_level(root);
        REQUIRE_THROWS_WITH(wrong_level.load_element("w", 0), Catch::Contains("level mismatch"));
        REQUIRE_THROWS_WITH(wrong_level.load_argument("w"), Catch::Contains("level mismatch"));
    }
}

TEST_CASE("manifest contracts reject geometry, level, shape and scale mismatches", "[parameter_manifest]") {
    const json tiles = {{"tile_grid", {4, 4}}, {"group_size", 16}, {"weight", true}};
    const json graph = {{"layer", {{"fc", {{"dense_seed_tiles", tiles}}}}}};
    json sig = {{"id", "densew_fc"}, {"size", {1, 3}}, {"level", 4}, {"type", "ct"},
                {"scale", 1099511627776.0}, {"layout", tiles}};
    const json signature = {{"encrypted_parameters", {sig}}};
    json arg = sig;
    arg["layer_id"] = "fc";
    json manifest = {{"arguments", {{"densew_fc", arg}}}};
    REQUIRE_NOTHROW(validate_parameter_manifest(manifest, signature, graph));
    auto reordered = manifest;
    reordered["arguments"]["densew_fc"]["layout"] = {
        {"weight", true}, {"group_size", 16}, {"tile_grid", {4, 4}}};
    REQUIRE_NOTHROW(validate_parameter_manifest(reordered, signature, graph));
    for (const char* field : {"size", "level", "type", "scale", "layout"}) {
        auto bad = manifest;
        if (std::string(field) == "size") bad["arguments"]["densew_fc"][field] = {1, 4};
        if (std::string(field) == "level") bad["arguments"]["densew_fc"][field] = 3;
        if (std::string(field) == "type") bad["arguments"]["densew_fc"][field] = "pt";
        if (std::string(field) == "scale") bad["arguments"]["densew_fc"][field] = 1.0;
        if (std::string(field) == "layout") bad["arguments"]["densew_fc"][field]["group_size"] = 8;
        REQUIRE_THROWS(validate_parameter_manifest(bad, signature, graph));
    }
    auto bad_graph = graph;
    bad_graph["layer"]["fc"]["dense_seed_tiles"]["tile_grid"] = {2, 8};
    REQUIRE_THROWS(validate_parameter_manifest(manifest, signature, bad_graph));
    manifest["arguments"]["densew_fc"].erase("layout");
    INFO("signature=" << signature.dump() << " manifest=" << manifest.dump());
    REQUIRE(signature.at("encrypted_parameters").at(0).contains("layout"));
    REQUIRE_FALSE(manifest.at("arguments").at("densew_fc").contains("layout"));
    REQUIRE_THROWS(validate_parameter_manifest(manifest, signature, graph));
    sig.erase("layout");
    sig.erase("scale");
    manifest["arguments"]["densew_fc"].erase("scale");
    REQUIRE_NOTHROW(validate_parameter_manifest(manifest, {{"offline", {sig}}}, graph));
}
