from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
import torch


class ModelForTripletExtraction(T5ForConditionalGeneration):

    def __init__(self, config):
        super(ModelForTripletExtraction, self).__init__(config)
        if hasattr(config, "_name_or_path"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                config._name_or_path)

        # self.to('cuda:1')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Get the configuration from the pretrained model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Load the model with the pretrained weights
        instance = super(ModelForTripletExtraction, cls).from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs)

        # Assign the tokenizer to the instance
        instance.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        return instance

    @classmethod
    def from_config(cls, config):
        # Create a new instance of ModelForTripletExtraction using the provided configuration
        instance = cls(config)

        # If the config has a "_name_or_path" attribute, load the tokenizer
        if hasattr(config, "_name_or_path"):
            instance.tokenizer = AutoTokenizer.from_pretrained(
                config._name_or_path)

        return instance

    # @classmethod
    # def from_pretrained_plus(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     # Get the configuration from the pretrained model
    #     config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    #     # Load the model with the pretrained weights
    #     instance = ModelForTripletExtraction.from_pretrained(
    #         pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    #     # Assign the tokenizer to the instance
    #     instance.tokenizer = AutoTokenizer.from_pretrained(
    #         pretrained_model_name_or_path)

    #     return instance

    # @classmethod
    # def from_config(cls, config):
    #     # Create a new instance of ModelForTripletExtraction using the provided configuration
    #     instance = cls(config)

    #     # If the config has a "_name_or_path" attribute, load the tokenizer
    #     if hasattr(config, "_name_or_path"):
    #         instance.tokenizer = AutoTokenizer.from_pretrained(
    #             config._name_or_path)

    #     return instance

    def generateFromText(self, prompt: str) -> str:
        # Encoding the input text
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Moving the tensor to GPU (if available)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            self.to("cuda")

        # Generating output
        output = self.generate(input_ids, max_new_tokens=100)

        # Decoding the output to get the text
        return self.tokenizer.decode(output[0])
    
    

# Example Usage:
# model = ModelForTripletExtraction.from_pretrained("path_or_name_of_your_pretrained_model")
# print(model.generateFromText("Your input text here"))
