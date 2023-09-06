# Model Related
from transformers import (AutoModelForCausalLM,AutoTokenizer,
                          AutoModelForSeq2SeqLM, T5Tokenizer, 
                          T5ForConditionalGeneration)
import openai, torch, os, re, random, string
# Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Data processing
import pandas as pd; import numpy as np
# progress visualization
from tqdm.notebook import tqdm; tqdm.pandas(); from tqdm import tqdm; tqdm.pandas()
# general packages
from typing import Union
from time import time
# from accelerate import infer_auto_device_map
# Statistics
from sklearn.model_selection import train_test_split
from datetime import datetime


class TFG:
        
        __DEVICE = 'cpu'
        __PARAMETERS = {'do_sample': False,
        'early_stopping': False,
        'max_length': 4,
        'min_length': 1,
        'num_beam_groups': 2,
        'num_beams': 2,
        #'max_tokens': 32,
        #'min_tokens': 1,
        'output_scores': True,
        'repetition_penalty': 1.0,
        'return_dict_in_generate': True,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0,}

        _SUPPORTED_MODELS = ('lmsys/vicuna-13b-v1.5-16k', 'lmsys/vicuna-13b-v1.5', 'lmsys/vicuna-7b-v1.5-16k',
                            'lmsys/longchat-7b-v1.5-32k', 'lmsys/vicuna-7b-v1.5', 'lmsys/vicuna-7b-v1.3', 
                            'lmsys/vicuna-13b-v1.3', 'lmsys/vicuna-7b-v1.1', 'lmsys/vicuna-13b-v1.1', 
                            'lmsys/vicuna-13b-delta-v0', 'lmsys/vicuna-7b-delta-v0', 'lmsys/vicuna-13b-delta-v1.1',
                            'lmsys/vicuna-7b-delta-v1.1', 'lmsys/vicuna-33b-v1.3', 'lmsys/longchat-13b-16k',
                            'lmsys/longchat-7b-16k', 'lmsys/fastchat-t5-3b-v1.0','google/flan-ul2',
                            'google/flan-t5-small','google/flan-t5-base','google/flan-t5-large',
                            'google/flan-t5-xl','google/flan-t5-xxl')

        __NO_INFO_ERROR = "Data Was Not Provided"
        _WRONG_DATA_ERROR = "Wrong Data Error"


        def __init__(self) -> object:
            """Initializes an instance of the TFG class.
               This class provides an interface for working with the T5-small model for conditional generation.
               This constructor results with a fast, minimal model for the current pipeline.

            Attributes:
                model_name (str): The name of the T5 model to be used.
                memory_saver (bool): If True, the model will be loaded in 8-bit precision to save memory.
                device (str): The device (e.g., 'cuda', 'cpu') on which the model will be loaded.
                parameters (dict): Additional parameters for model configuration.
                api_model (bool): If True, indicates that an API endpoint will be used for the model.
                api_key (bool): If True, an API key is required for accessing the API endpoint.
                api_base (str): The base URL for the API endpoint.
                model_dev (str): The development version of the model extracted from the model_name.
                model_version (str): The version of the model extracted from the model_name.
                tokenizer (T5Tokenizer): Tokenizer for processing text inputs for the model.
                model (T5ForConditionalGeneration): The T5 model for conditional generation.

            Returns:
                object: work pipeline for social science experiments, supported by a flan-t5-small.
            """
            self._model_name = 'google/flan-t5-small'
            self._memory_saver = False
            self._device = self.__DEVICE
            self._parameters = self.__PARAMETERS
            self._api_model = False
            self._api_key = False
            self._api_base = None
            self._model_dev = _Tools._split_dev_ver(input_string=self._model_name)[0]
            self._model_version = _Tools._split_dev_ver(input_string=self._model_name)[1]
            self.tokenizer = T5Tokenizer.from_pretrained(self._model_name)
            print(f'{self._model_name} tokenizer loaded. to device: device= {self._device}')
            self.model = T5ForConditionalGeneration.from_pretrained(self._model_name, load_in_8bit = self._memory_saver).to(self._device)

        def __init__(self, model_name :str, api_base :str = None, configuration :dict = __PARAMETERS,
                     connect_to_gpu :bool = True, memory_saver :bool = False, api_model :bool = False,
                     api_key :str = "EMPTY", model_out_of_pipline :object = None, tokenizer_out_of_pipline :object = None,
                     model_developer :str = None, model_version :str = None) -> object:
            """Initializes an instance of the TFG class.
               This class provides an interface for working with a model of choice from the pool, for text generation.
               This constructor allows felxibility with the input model for the current pipeline.

            Args:
                model_name (str): The name of the model to be used.
                api_base (str, optional): The base URL for the API endpoint. Default is None.
                configuration (dict, optional): Additional configuration parameters. Default is __PARAMETERS.
                connect_to_gpu (bool, optional): If True, the model will be loaded on a GPU if available. Default is True.
                memory_saver (bool, optional): If True, the model will be loaded with memory-saving settings. Default is False.
                api_model (bool, optional): If True, indicates that an API endpoint will be used for the model. Default is False.
                api_key (str, optional): The API key for accessing the API endpoint. Default is "EMPTY".
                model_out_of_pipeline (object, optional): Pre-loaded model instance to use. Default is None.
                tokenizer_out_of_pipeline (object, optional): Pre-loaded tokenizer instance to use. Default is None.
                model_developer (str, optional): The developer of the model. Default is None.
                model_version (str, optional): The version of the model. Default is None.
    
            Returns:
                object: An instance of the class.
            """
            self._model_name = model_name
            _Tools._supported_model(self)
            self._memory_saver = memory_saver
            self._connect_to_gpu = connect_to_gpu
            if configuration == None:
                self._parameters = self.__PARAMETERS
            else:
                self._parameters = configuration
            self._api_model = api_model
            self._api_key = api_key
            self._api_base = api_base
            try:
                if model_developer == None:
                    self._model_dev = _Tools._split_dev_ver(input_string=self._model_name)[0]
            except:
                self._model_dev = self.__NO_INFO_ERROR
            try:
                if model_version == None:
                    self._model_version = _Tools._split_dev_ver(input_string=self._model_name)[1]
            except:
                self._model_version = self.__NO_INFO_ERROR
            if (model_out_of_pipline is not None) and (tokenizer_out_of_pipline is not None):
                self.model = model_out_of_pipline
                self.tokenizer = tokenizer_out_of_pipline
                _Tools._gpu_check_and_connect(self)
            elif self._api_model or api_base:
                 print(f'constructor api_model {self._api_model}')
                 print(f'With {self._model_name}, connecting to {self._api_base}. API key = {self._api_key}')
                 self._device = None
            else:
                _Tools._gpu_check_and_connect(self)
                if self._model_dev == "google":
                    if self._model_version == "flan-ul2":
                        _Tools._ul2_loader(self)
                    else:
                        _Tools._t5_loader(self)
                else:
                    _Tools._vicuna_and_auto_loader(self)
        
        def get_model_developer(self) -> str:
            """Get the developer of the model.
        
            Returns:
                str: The developer of the model.
            """
            return self._model_dev
        
        def get_model_name(self) -> str:
            """Get the name of the model.

            Returns:
                str: The model name.
            """
            return self._model_name
        
        def get_model_version(self) -> str:
            """Get the version of the model.

            Returns:
                str: The version of the model.
            """
            return self._model_version
        
        def get_device_status(self) -> str:
            """Get the status of the device.

            Returns:
                str: The device in use.
            """
            return self._device
        
        def get_config(self) -> dict:
            """Get the configuration parameters of the instance.
            
            Returns:
                dict: A dictionary containing the configuration parameters.
            """
            return self._parameters
        
        def set_model_developer(self, developer :str):
            """Set the developer of the model.

            Args:
                developer (str): The new developer name.

            Returns:
                    None
            """
            self._model_dev = developer
            print(f'New model developer set')
        
        def set_model_name(self, model_full_name :str):
            """Set the full name of the model.
            
            Args:
                model_full_name (str): The new full name of the model.
            Returns:
                    None
            """
            self._model_name = model_full_name
            print(f'New model full name set')
        
        def set_model_version(self, model_version :str):
            """Set the version of the model.

            Args:   
                model_version (str): The new version of the model.

            Returns:        
                    None
            """
            self._model_version = model_version
            print(f'New model version set')
        
        def set_device_status(self, device_name :str):
            """Set the device status of the instance.

            Args:
                device_name (str): The new device status.
            
            Returns:
                    None
            """
            self._device = device_name
            print(f'New device set')
        
        def set_config(self, new_config :dict):
            """Set the config configuration parameters of the instance

            Args:
                new_config (dict): The new dictonary containing the configuration parameters.
            """
            self._parameters = new_config
            print(f'New config set')
        
        def get_tokenizer(self) -> Union[AutoTokenizer,T5Tokenizer]:
            """Get the tokenizer used by the instance.
        
            Returns:
                Union[AutoTokenizer,T5Tokenizer]: The tokenizer instance.
            """
            return self.tokenizer
        
        def get_model(self) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration]:
            """Get the model used by the instance.

            Returns:
                Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration]: The model instance.
            """
            return self.model
        
        def get_api_model(self) -> bool:
            """Get the status of the API model.

            Returns:
                bool: True if the instance is an API model, False otherwise.
            """
            return self._api_model
        
        def get_api_key(self) -> str:
            """Get the API key.

            Returns:
                str: The API key associated with the instance.
            """
            return self._api_key
        
        def get_api_base(self) -> str:
            """Get the API base.

            Returns:
                str: The API base associated with the instance.
            """
            return self._api_base
        
        def get_model_settings_df(self) -> pd.DataFrame:
            """Get the model settings as a DataFrame.

            Returns:
                pd.DataFrame: A DataFrame containing the model settings.
            """
            return _Tools._get_model_settings(TFG=self)


        def generate(self, prompt, text_only = False) -> Union[str, dict]:
            """Generate a response using the TFG instance.

            Args:
                prompt (str): The input prompt for generation.
                text_only (bool, optional): If True, only the generated text is returned;
                                            otherwise, the full generation details are returned.

            Returns:
                Union[str, dict]: Depending on the 'text_only' parameter, either a dictionary
                                containing the generation details or the generated text.
            """
            return _Generation._generate_from_tfg(self, prompt=prompt, text_only=text_only)
        
        def generate_for_ds(self, df :pd.DataFrame, text_only :bool):
            """Generate responses for a DataFrame using the instance.

            Args:
                df (pd.DataFrame): The DataFrame containing input prompts.
                text_only (bool): If True, only the generated text is added to the DataFrame.

            Returns:
                pd.DataFrame: The DataFrame with generated responses.
            """
            return Datasets._generate_for_tfg_ds(self,df=df,text_only=text_only)

        def decode_with_instance(self, outputs) -> str:
            """Decode model outputs, converts sequence of ids in a string, using the tokenizer.

            Args:
                TFG (object): The instance
                outputs (_type_): The model outputs to be decoded.

            Returns:
                str: The decoded text.
            """
            return self.get_tokenizer().decode(outputs)
            
        def encode_with_instance(self, inputs :str) -> torch.Tensor:
            """Encode input text into a tensor using the tokenizer.

            Args:
                TFG (object): The instance
                inputs (_type_): The input text to be encoded.

            Returns:
                torch.Tensor: The encoded tensor representation of the input text.
            """
            return self.get_tokenizer().encode(inputs)
            

class Datasets(TFG):
    
    def _dataset_creator_from_df(questionnaire_name :str, df :pd.DataFrame, female_ratio) -> pd.DataFrame:
        """Creates a dataset for the pipeline, allows to adjust female/male ratio of the sample and assign it randomly.
        The input pd.DataFrame should be with the following order:
        XXXXXXXX
        The new dataframe will include the following columns:
        Index(['questionnaire', 'q_item', 'title', 'surname', 'racial_group',
       'full_prompt', 'generated text', 'scores', 'outputs'],
        dtype='object')
        When the configuration of the model is set to ('output_scores': False) or when generation is set to (text_only = True),
        the 'scores' and 'outputs' columns will not show.

        Args:
            questionnaire_name (str): The questionnaire name.
            df (pd.DataFrame): The questionnaire in the required format.
            female_ratio (_type_): The desired female-to-male ratio in the dataset (the ratio of title Ms. in the dataset).

        Returns:
            pd.DataFrame: The new adjusted dataset for the pipeline.
        """
        df_names = pd.read_csv('/home/pop536648/datasets/names_new.csv', index_col=0).sample(frac=1) # change to something accessible 
        new_df = pd.DataFrame()
        male, female = train_test_split(df_names, test_size=female_ratio, random_state=0)
        fixed_prompt = "was asked the following question."    
        pbar = tqdm(["Male", "Female"])
        for char in pbar:
            pbar.set_description("Processing %s" % char)
            for i in male.iterrows():
                for d in df.iterrows():
                    temp_dict = {"questionnaire":questionnaire_name,
                                "q_number":d[0]+1,
                                "q_item":d[1].Qs,
                                "title": "Mr.",
                                "surname":i[1].names,
                                "racial_group":i[1].racial_group}
                    temp_dict["full_prompt"] = f"{temp_dict['title']} {temp_dict['surname']} {fixed_prompt} {temp_dict['q_item']}"
                    new_df = pd.concat([new_df, pd.DataFrame(temp_dict,index=[0])], ignore_index = True)
                    
            for i in female.iterrows():
                for d in df.iterrows():
                    temp_dict = {"questionnaire":questionnaire_name,
                                "q_number":d[0]+1,
                                "q_item":d[1].Qs,
                                "title": "Ms.",
                                "surname":i[1].names,
                                "racial_group":i[1].racial_group}
                    temp_dict["full_prompt"] = f"{temp_dict['title']} {temp_dict['surname']} {fixed_prompt} {temp_dict['q_item']}"
                    new_df = pd.concat([new_df, pd.DataFrame(temp_dict,index=[0])], ignore_index = True)
        #new_df = new_df.sample(frac=1)
        print(f'Male sample size: {len(male)*len(df)}.\nFemale sample size: {len(female)*len(df)}.')
        return new_df
    
    def dataset_creator(questionnaire_name :str, full_path :str = None, df :pd.DataFrame = None, female_ratio = 0.5) -> pd.DataFrame:
        """Create a dataset from a CSV file or an existing DataFrame.
        
        Args:
            questionnaire_name (str): The name of the questionnaire for the dataset.
            full_path (str, optional): The full path to a CSV file containing data. Default is None.
            df (pd.DataFrame, optional): An existing DataFrame containing data. Default is None.
            female_ratio (_type_): The desired female-to-male ratio in the dataset (the ratio of title Ms. in the dataset). Default is 0.5.

        Returns:
            pd.DataFrame: The created dataset
        """
        if full_path is not None:
            df = pd.read_csv(full_path)
        return Datasets._dataset_creator_from_df(df=df, female_ratio = female_ratio, questionnaire_name=questionnaire_name)           

    def _generate_for_tfg_ds(TFG, df :pd.DataFrame, text_only :bool = None) -> pd.DataFrame:
        """Generate responses for a DataFrame using the TFG instance.

        Args:
            TFG: The TFG instance for generation.
            df (pd.DataFrame): The DataFrame containing input prompts.
            text_only (bool, optional): If True, only the generated text is added to the DataFrame.
                                        If False and TFG outputs scores, generated text, scores, and outputs are added.
                                        If None, the behavior is based on TFG configuration. Default is None.
        Returns:
                pd.DataFrame: The DataFrame with generated responses.
        """
        df_n = df.copy()
        if "full_prompt" in df_n.columns:
            generetad_lst = []
            for d in tqdm(df_n.iterrows(), total = len(df_n)):
                generetad_lst.append(TFG.generate(prompt=d[1]['full_prompt']))
            if text_only == True or TFG._parameters['output_scores'] == False:
                if TFG._parameters['output_scores'] and isinstance(generetad_lst[0], tuple):
                    outputs_lst = list(map(list, zip(*generetad_lst)))
                    df_n['generated text'] = outputs_lst[0]
                else:
                    df_n['generated text'] = generetad_lst
            else:
                outputs_lst = list(map(list, zip(*generetad_lst)))
                df_n['generated text'] = outputs_lst[0]
                df_n['scores'] = outputs_lst[1]
                df_n['outputs'] = outputs_lst[2]
                return df_n
        else:
            print(f'{TFG._WRONG_DATA_ERROR}')
        return df_n
                

    class Analysis:
        
            @staticmethod
            def get_sentiment(prompt :str, verbal_score = False) -> Union [float,str]:
                """The functions gets an input prompt and returns the sentiment score by the following scale:
                negative sentiment: score <= -0.5
                neutral sentiment:  -0.5 < score < 0.5
                positive sentiment: 0.5 <= score

                Args:
                    prompt (str): A string that holds a word or a sentance.
                    verbal_score (bool, optional): If True, returns a verbal label ('pos', 'neu', 'neg').
                                       If False, returns the sentiment compound score.

                Returns:
                    Union[float, str]: Depending on the 'verbal_score' parameter, either a sentiment score or a verbal label.
                """
                analyzer = SentimentIntensityAnalyzer()
                if verbal_score:
                    score = analyzer.polarity_scores(prompt)['compound']
                    if score <= -0.5:
                        return "neg"
                    elif -0.5 < score and score < 0.5:
                        return "neu"
                    else:
                        return "pos"
                else:
                    return analyzer.polarity_scores(prompt)['compound']
            
            @staticmethod
            def prompt_descriptives(prompt :str, filter_non_alphabetic = True) -> dict:
                """Calculate descriptive statistics for the lengths of words in the input prompt.

                Args:
                    prompt (str): The input prompt for analysis.
                    filter_non_alphabetic (bool, optional): If True, non-alphabetic characters are filtered out before analysis. Default is True.

                Returns:
                    dict: A dictionary containing the maximum, minimum, and average lengths of words.
                """
                if filter_non_alphabetic:
                    words = _Tools._remove_non_alphabetic(prompt).split()
                else:
                    words = prompt.split()
                d = {"Max length": len(max(words, key=len)),
                    "Min length": len(min(words, key=len)),
                    "Average length": round(sum(map(len, words))/float(len(words)),3)}
                return d
            
class _Tools(TFG):
            
            def _gpu_check_and_connect(TFG):
                """Check for GPU availability and set the appropriate device for the TFG instance.

                Args:
                    TFG (object): The instance
                """
                if TFG._connect_to_gpu:
                    if torch.cuda.is_available():
                        TFG._device = torch.device('cuda')
                        print(f'There are {torch.cuda.device_count()} GPU(s) available.\nDevice name: {torch.cuda.get_device_name(0)}')
                    else:
                        print('No GPU available, using the CPU instead.')
                        TFG._device = torch.device("cpu")
                else:
                    TFG._device = torch.device("cpu")
                    print(f'* {TFG._device} in use *'.upper())

            def _t5_loader(TFG):
                """Load a T5 model and tokenizer for the TFG instance.


                Args:
                    TFG (_type_): _description_
                """
                TFG.tokenizer = T5Tokenizer.from_pretrained(TFG._model_name)
                print(f'{TFG._model_name} - Tokenizer loaded')
                if TFG._connect_to_gpu:
                    TFG.model = T5ForConditionalGeneration.from_pretrained(TFG._model_name, device_map="auto", load_in_8bit = TFG._memory_saver)
                else:
                    # KATE: commenting this out bc the load_in8bit argument was throwing unexpected keyword argument 'load_in_8bit'. Come back to this # todo
                    # TFG.model = T5ForConditionalGeneration.from_pretrained(TFG._model_name, load_in_8bit = TFG._memory_saver).to(TFG._device)
                    TFG.model = T5ForConditionalGeneration.from_pretrained(TFG._model_name,).to(TFG._device)
                print(f'T5 loader done, {TFG._model_name} - Model loaded to {TFG._device}')

            def _ul2_loader(TFG):    
                """Load a ul2-flan model and tokenizer for the TFG instance.

                Args:
                    TFG (object): The instance
                """
                TFG.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
                print(f'{TFG._model_name} - Tokenizer loaded')
                if TFG._connect_to_gpu:
                    TFG.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit = TFG._memory_saver)
                else:
                    TFG.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", load_in_8bit = TFG._memory_saver).to(TFG._device)
                print(f'UL2 loader done, {TFG._model_name} - Model loaded to {TFG._device}')

            def _vicuna_and_auto_loader(TFG):
                """Load a Transformer and tokenizer for the TFG instance, the function automatically loads a fast tokenizer version when available. 

                Args:
                    TFG (object): The instance
                """
                try:
                    TFG.tokenizer = AutoTokenizer.from_pretrained(TFG._model_name, use_fast = True, trust_remote_code=True)
                except:
                    TFG.tokenizer = AutoTokenizer.from_pretrained(TFG._model_name, use_fast = False, trust_remote_code=True)
                print(f'{TFG._model_name} - Tokenizer loaded')
                if TFG._connect_to_gpu:
                    TFG.model = TFG.model = AutoModelForCausalLM.from_pretrained(TFG._model_name, trust_remote_code=True, device_map="auto", load_in_8bit = TFG._memory_saver)
                else:
                    TFG.model = AutoModelForCausalLM.from_pretrained(TFG._model_name, trust_remote_code=True, load_in_8bit = TFG._memory_saver).to(TFG._device)
                print(f'Auto Loader done, {TFG._model_name} - Model loaded to {TFG._device}')

            def _supported_model(TFG):
                """Check if the selected model is supported by the current pipeline.

                Args:
                    TFG (object): The instance
                """
                model_name_to_check = TFG.get_model_name()
                if model_name_to_check in TFG._SUPPORTED_MODELS:
                    print(f'The model {model_name_to_check} is supported by this pipeline')
                else:
                    print(f'The model {model_name_to_check} is *NOT* supported by this pipeline, please check the following:\n{TFG._SUPPORTED_MODELS}')        

            @staticmethod
            def _split_dev_ver(input_string) -> tuple:
                """Split the input string into developer and version components.
                
                Args:
                    input_string (str): The input string to be split.

                Returns:
                    tuple: A tuple containing the developer and version components, or False if the pattern doesn't match.
                """
                pattern = r'^(.*?)/(.*)$'
                match = re.match(pattern, input_string)
                if match:
                    return match.group(1), match.group(2)
                else:
                    return False
                
            @staticmethod
            def _remove_non_alphabetic(prompt) -> str:
                """Remove non-alphabetic characters from the input prompt.

                Args:
                    prompt (str): The input prompt containing words.

                Returns:
                    str: The input prompt with non-alphabetic characters removed.
                """
                words = prompt.split()
                return " ".join([word for word in words if word.isalpha()])
            

            def _get_model_settings(TFG):# -> pd.DataFrame:
                #print(TFG.get_config())
                data = {'model_name': TFG.get_model_name(),
                        'model_dev': TFG.get_model_developer(),
                        'model_ver': TFG.get_model_version(),
                        'device': TFG.get_device_status(),
                        'config': [TFG.get_config()],
                        'api_model': TFG.get_api_model(),
                        'api_key': TFG.get_api_key(),
                        'api_base': TFG.get_api_base(),
                        'tokenizer': TFG.get_tokenizer(),
                        'model': TFG.get_model(),
                        'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                df = pd.DataFrame(data, index=[0])
                return df

class _Generation(TFG):
      
            def _generation_without_score(TFG, prompt :str) -> str:
                """Generate a response without scoring.

                Args:
                    TFG (object): The instance
                    prompt (str): The input prompt for generation.                    

                Returns:
                    str: The generated response.
                """
                input_ids = TFG.tokenizer(prompt, return_tensors="pt").input_ids.to(TFG._device)
                output = TFG.model.generate(input_ids, **TFG._parameters)
                return TFG.tokenizer.decode(output['sequences'][0])

            def _generation_with_score(TFG, prompt :str) -> tuple:
                """Generate a response with scoring, and full output.
                
                Args:
                    TFG (object): The instance
                    prompt (str): The input prompt for generation.

                Returns:
                    tuple: A tuple containing three lists - generated responses, scores, and output details.
                """
                input_ids = TFG.tokenizer(prompt, return_tensors="pt").input_ids.to(TFG._device)
                output = TFG.model.generate(input_ids, **TFG._parameters)
                answer = TFG.tokenizer.decode(output['sequences'][0])
                if TFG._parameters['num_beams'] > 1:
                    score = float(torch.sigmoid(output['sequences_scores'][0]))
                else:
                    score = 0        
                return answer, score, output
            
            def _generate_api(TFG, prompt, text_only) -> Union[dict, str]:
                """Generate a response using an API.
                
                Args:
                    TFG (object): The instance
                    prompt (str): The input prompt for generation.
                    text_only (bool): If True, only the generated text is returned; otherwise, the full API response is returned.

                Returns:
                    Union[dict, str]: Depending on the 'text_only' parameter, either a dictionary containing the API response or the generated text.
                """
                openai.api_key = TFG._api_key
                openai.api_base = TFG._api_base
                if text_only:
                    return openai.Completion.create(model=TFG._model_version, prompt=prompt, **TFG._parameters)['choices'][0]['text']
                else:
                    return openai.Completion.create(model=TFG._model_version, prompt=prompt, **TFG._parameters)
            
            def _generate_from_tfg(TFG, prompt :str, text_only :bool) -> Union[dict, str]:
                """Generate a response based on the TFG instance configuration. 

                Args:
                    TFG (object): The instance
                    prompt (str): The input prompt for generation.
                    text_only (bool): If True, only the generated text is returned; otherwise, the full generation output is returned.

                Returns:
                    Union[dict, str]: Depending on the instance configuration and 'text_only' parameter, either a dictionary containing the generation output or the generated text
                """
                if TFG._api_model:
                    openai.api_key = TFG._api_key
                    openai.api_base = TFG._api_base
                    return _Generation._generate_api(TFG, prompt=prompt, text_only=text_only)
                if TFG._parameters['output_scores'] and not text_only:
                    return _Generation._generation_with_score(TFG, prompt=prompt)
                else:
                    return _Generation._generation_without_score(TFG, prompt=prompt)
                