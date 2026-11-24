// Secret-key-assisted causal diagnostic. Never part of the inference runtime.
TEST_CASE("cpu/maxpool_p1_oracle_control", "[maxpool_p1_control][diagnostic]") {
    const char* configured = std::getenv("LATTI_MAXPOOL_P1_CONTROL_DIR");
    REQUIRE(configured != nullptr);
    const fs::path root(configured);
    const auto config = read_json((root / "control_config.json").string());
    REQUIRE(config.at("alpha").get<int>() == 14);
    const bool bootstrap_input = config.value("bootstrap_input", false);
    const auto expected = read_json((root / "ckks_parameter.json").string()).at("param0");
    const bool custom_precision = config.value("custom_precision", false);
    auto scheme_param = CkksParameter::create_custom_parameter(
        static_cast<int>(std::log2(expected.at("poly_modulus_degree").get<int>())),
        expected.at("log_default_scale").get<int>(),
        expected.at("q").get<std::vector<uint64_t>>(),
        expected.at("p").get<std::vector<uint64_t>>(),
        expected.value("ring_type", std::string("standard")) == "conjugate_invariant"
            ? RingType::ConjugateInvariant : RingType::Standard);
    auto context = CkksContext::create_random_context(scheme_param, EncryptorType::PUBLIC_KEY, bootstrap_input);
    auto decode_pt = [&](const CkksPlaintext& pt) {
        std::vector<double> values;
        context.decode(pt, values);
        return values;
    };
    std::vector<double> input;
    std::ifstream input_file(root / "input.csv");
    double value;
    while (input_file >> value) input.push_back(value);
    REQUIRE(input.size() == size_t(scheme_param.max_slots()));
    const auto coefficients = config.at("coefficients").get<std::vector<double>>();
    const auto scales = config.at("coefficient_scales").get<std::vector<double>>();
    const auto coefficient_levels = config.at("coefficient_levels").get<std::vector<int>>();
    const auto levels = config.at("baby_levels").get<std::vector<int>>();
    const double default_scale = config.at("default_scale");
    if (!custom_precision) REQUIRE(default_scale == scheme_param.default_scale());
    const int degree = config.value("degree", 27);
    REQUIRE((degree == 15 || degree == 27 || degree == 29));
    REQUIRE(coefficients.size() == size_t(degree+1));
    REQUIRE(scales.size() == coefficients.size());
    REQUIRE(coefficient_levels.size() == coefficients.size());
    REQUIRE(levels.size() == size_t((degree+4)/4));
    const int input_level = config.at("input_level");
    auto x = context.encrypt(fhe_api::encode(context, input, input_level, config.at("input_scale").get<double>()));
    auto decode = [&](const CkksCiphertext& ct) { return decode_pt(context.decrypt(ct)); };
    auto dump = [&](const std::string& name, const std::vector<double>& values) {
        std::ofstream file(root / (name + ".csv"));
        file.precision(17);
        for (double v : values) file << v << '\n';
    };
    dump("pre_bootstrap_input", decode(x));
    nlohmann::json bootstrap_metadata;
    if (bootstrap_input) {
        REQUIRE_FALSE(custom_precision);
        REQUIRE(input_level == 5);
        x = context.drop_level(x, x.level());
        std::cout << "[P1 control] real bootstrap" << std::endl;
        x = context.bootstrap(x);
        REQUIRE(x.level() == input_level);
        REQUIRE(std::abs(x.scale()/default_scale-1) < 1e-12);
        bootstrap_metadata = {{"level", x.level()}, {"scale", x.scale()}};
    }
    dump("actual_input", decode(x));
    std::vector<CkksPlaintext> weights;
    std::vector<CkksPlaintext> full_weights;
    nlohmann::json coefficient_checks = nlohmann::json::array();
    const bool full_rns = config.value("coefficient_encoding", std::string("ringt")) == "full_rns";
    for (size_t i = 0; i < coefficients.size(); ++i)
        weights.push_back(fhe_api::encode_ringt(context, std::vector<double>(input.size(), coefficients[i]), scales[i]));
    for (size_t i = 0; i < coefficients.size(); ++i) {
        auto decoded = decode_pt(weights[i]);
        double ringt_error = 0.;
        for (double v : decoded) ringt_error = std::max(ringt_error, std::abs(v-coefficients[i]));
        nlohmann::json check = {{"index", i}, {"coefficient", coefficients[i]},
            {"scale", scales[i]}, {"ringt_max_error", ringt_error}};
        if (full_rns) {
            full_weights.push_back(fhe_api::encode(context, std::vector<double>(input.size(), coefficients[i]),
                                                  coefficient_levels[i], scales[i]));
            decoded = decode_pt(full_weights.back());
            double full_error = 0.;
            for (double v : decoded) full_error = std::max(full_error, std::abs(v-coefficients[i]));
            check["full_rns_max_error"] = full_error;
            REQUIRE(full_error < 1e-10 * std::max(1., std::abs(coefficients[i])));
        }
        coefficient_checks.push_back(check);
    }
    auto align = [&](const CkksCiphertext& ct, int level) {
        REQUIRE(ct.level() >= level);
        return ct.level() == level ? ct.copy() : context.drop_level(ct, ct.level()-level);
    };
    auto checked_rescale = [&](const CkksCiphertext& raw) {
        REQUIRE(raw.level() > 0);
        const double expected_scale = raw.scale()/scheme_param.q()[raw.level()];
        auto result = context.rescale(raw);
        REQUIRE(result.level() == raw.level()-1);
        REQUIRE(std::abs(result.scale()/expected_scale-1) < 1e-12);
        return result;
    };
    auto mul_weight = [&](const CkksCiphertext& ct, int i) {
        REQUIRE(ct.level() == coefficient_levels.at(i));
        if (full_rns)
            return checked_rescale(context.mult(ct, full_weights.at(i)));
        const auto& pt = weights.at(i);
        return checked_rescale(context.mult(ct, pt));
    };
    auto product = [&](const CkksCiphertext& a, const CkksCiphertext& b) {
        return checked_rescale(context.relinearize(context.mult(a,b)));
    };
    nlohmann::json metadata;
    if (bootstrap_input) metadata["bootstrap"] = bootstrap_metadata;
    const auto modes = config.value("modes", std::vector<std::string>{
        "baseline", "zero_control", "correct_x4_rescale", "correct_x4_local", "correct_x2_local"});
    for (const std::string& mode : modes) {
        REQUIRE((mode == "baseline" || mode == "zero_control" || mode == "correct_x4_rescale" ||
                 mode == "correct_x4_local" || mode == "correct_x2_local"));
        std::cout << "[P1 control] " << mode << std::endl;
        std::map<int, CkksCiphertext> powers;
        powers.emplace(1, x.copy());
        const auto square_powers = config.value("square_powers", std::vector<int>{2,4,8,16});
        for (int power : square_powers) {
            const auto& operand = powers.at(power/2);
            auto raw = context.relinearize(context.mult(operand,operand));
            auto result = checked_rescale(raw);
            bool correct = (power == 4 && mode != "baseline" && mode != "correct_x2_local") ||
                           (power == 2 && mode == "correct_x2_local");
            if (correct) {
                auto decoded = decode(result);
                std::vector<double> correction(decoded.size(), 0.);
                if (mode != "zero_control") {
                    auto target = mode == "correct_x4_rescale" ? decode(raw) : decode(operand);
                    for (size_t i = 0; i < correction.size(); ++i) {
                        const long double desired = mode == "correct_x4_rescale" ? target[i] :
                            (long double)target[i]*target[i];
                        correction[i] = double(desired-decoded[i]);
                    }
                }
                // Oracle correction depends on the secret key in this test only.
                auto pt = fhe_api::encode(context, correction, result.level(), result.scale());
                result = context.add(result, pt);
            }
            dump(mode+"_power_"+std::to_string(power), decode(result));
            powers.emplace(power, std::move(result));
        }
        std::vector<CkksCiphertext> nodes;
        for (size_t g = 0; g < levels.size(); ++g) {
            int base = 4*g, level = levels[g];
            auto linear = mul_weight(align(x, level+1), base+1);
            auto sum = linear.copy();
            if (base+2 <= degree) {
                auto quadratic = mul_weight(align(powers.at(2), level+1), base+2);
                sum = context.add(sum,quadratic);
            }
            if (base+3 <= degree) {
                auto cubic_linear = mul_weight(align(x, level+2), base+3);
                auto cubic = product(cubic_linear, align(powers.at(2), level+1));
                sum = context.add(sum,cubic);
            }
            nodes.push_back(full_rns ? context.add(sum,full_weights.at(base)) :
                                     context.add(sum,weights.at(base)));
        }
        int combine = 4;
        while (nodes.size() > 1) {
            std::vector<CkksCiphertext> next;
            for (size_t i = 0; i < nodes.size(); i += 2) {
                if (i+1 == nodes.size()) { next.push_back(nodes[i].copy()); continue; }
                int level = nodes[i].level();
                auto term = product(align(nodes[i+1], level+1), align(powers.at(combine), level+1));
                next.push_back(context.add(nodes[i],term));
            }
            nodes = std::move(next);
            combine *= 2;
        }
        REQUIRE(nodes[0].level() == config.value("output_level", 0));
        REQUIRE(input_level - nodes[0].level() == config.value("level_cost", 5));
        REQUIRE(std::abs(nodes[0].scale()/default_scale-1) < 1e-12);
        const auto values = decode(nodes[0]);
        for (double v : values) REQUIRE(std::isfinite(v));
        dump(mode+"_output", values);
        metadata[mode] = {{"level",nodes[0].level()},{"scale",nodes[0].scale()}};
        if (config.value("binary_finish", false)) {
            REQUIRE(mode == "baseline");
            REQUIRE(nodes[0].level() >= 1);
            auto load = [&](const std::string& name) {
                std::ifstream file(root / (name+".csv"));
                std::vector<double> data;
                double v;
                while (file >> v) data.push_back(v);
                REQUIRE(data.size() == input.size());
                return data;
            };
            const int level = nodes[0].level();
            auto difference = context.encrypt(fhe_api::encode(context, load("binary_difference"), level, default_scale));
            auto summation = context.encrypt(fhe_api::encode(context, load("binary_sum"), level, default_scale));
            auto term = product(difference,nodes[0]);
            auto result = context.add(align(summation,term.level()),term);
            REQUIRE(result.level() == level-1);
            REQUIRE(std::abs(result.scale()/default_scale-1) < 1e-12);
            auto finish_values = decode(result);
            for (double v : finish_values) REQUIRE(std::isfinite(v));
            dump("binary_finish_output",finish_values);
            metadata["binary_finish"] = {{"level", result.level()}, {"scale",result.scale()}};
        }
    }
    metadata["warning"] = custom_precision ?
        "Isolated precision diagnostic; no bootstrap or full MaxPool acceptance" :
        "Oracle diagnostic, not a deployable fix or accuracy acceptance test";
    metadata["coefficient_checks"] = coefficient_checks;
    std::ofstream(root / "control_metadata.json") << metadata.dump(2);
}
