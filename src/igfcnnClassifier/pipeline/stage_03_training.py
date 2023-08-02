from src.igfcnnClassifier.config.configuration import ConfigurationManager
from src.igfcnnClassifier.components.prepare_model import PrepareBaseModel

# from src.igfcnnClassifier.components.prepare_callbacks import PrepareCallback
from src.igfcnnClassifier.components.model_training import Training
from src.igfcnnClassifier import logger

# import sys
# sys.path.append("/home/vinayak.t/IGF-CNN-research-deployment/src/igfcnnClassifier/components/")
# from model_training import Training

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        # prepare_callbacks_config = config.get_prepare_callback_config()
        # prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        # callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        
        model_config = config.get_prepare_base_model_config()
        training_config = config.get_training_config()
        training = Training(config=training_config,model=PrepareBaseModel,config1=model_config)
        # training.get_base_model()
        training.train_valid_generator()
        training.train()
        # training.train(
        #     callback_list=callback_list
        


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        