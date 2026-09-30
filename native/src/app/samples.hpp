// Built-in sample parts with ready-made study setups.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/mesh.hpp"
#include "fea/structural.hpp"

namespace ps {

struct SampleSetup {
    std::vector<Fixture> fixtures;
    std::vector<Load> loads;
    std::string material;
    struct Air { double yaw, pitch, speed, ground = -1; };  // ground: clearance to a moving road (model units), < 0 free air
    std::optional<Air> airflow;
};

struct Sample {
    std::string id, name, note;
    std::function<MeshSource()> make;
    std::function<SampleSetup(const Part&)> setup;
};

const std::vector<Sample>& samples();
const Sample* findSample(const std::string& id);

}  // namespace ps
