from src.igfcnnClassifier.config.configuration import ConfigurationManager
from src.igfcnnClassifier.components.prepare_model import PrepareBaseModel
from src.igfcnnClassifier import logger
# import sys
# sys.path.append("/home/vinayak.t/IGF-CNN-research-deployment/src/igfcnnClassifier/components/")
# import prepare_base_model


STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.update_base_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e