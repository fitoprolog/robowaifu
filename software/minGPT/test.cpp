#include <torch/script.h>

int main() {
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("path/to/module.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of input tensors.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and get the output.
    at::Tensor output = module.forward(inputs).toTensor();

    // Print the output.
    std::cout << output << '\n';

    return 0;
}
