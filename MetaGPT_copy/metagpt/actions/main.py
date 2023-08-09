from metagpt.actions.model_training import ModelTraining
from metagpt.actions.code_monitoring import CodeMonitoring
from metagpt.actions.code_deployment import CodeDeployment
from metagpt.actions.model_evaluation import ModelEvaluation
from metagpt.actions.data_loading import DataLoader
from metagpt.actions.model_inference import ModelInference


def main():
    # Initialize actions
    model_training = ModelTraining()
    code_monitoring = CodeMonitoring()
    code_deployment = CodeDeployment()
    model_evaluation = ModelEvaluation()
    data_loading = DataLoader()
    model_inference = ModelInference()

    # Perform actions
    model_training.train_model()
    code_monitoring.monitor_code()
    code_deployment.deploy_code()
    model_evaluation.evaluate_model()
    data_loading.load_data()
    model_inference.infer_model()


if __name__ == '__main__':
    main()