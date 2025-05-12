"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="End2EndBirdmodelgroup",
    pipeline_name="End2EndbirdPipeline",
    base_job_prefix="End2Endbirdbird",
    processing_instance_type="ml.m5.large",
    training_instance_type="ml.m5.large",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="https://sagemaker-us-east-1-729987989507.s3.amazonaws.com/bird-groundtruth/unlabeled/images/",
    )
    
    
    # we also have input_maifest to we will add it here
    input_manifest = ParameterString(
        name="InputManifestUrl",
        default_value="https://sagemaker-us-east-1-729987989507.s3.amazonaws.com/bird-groundtruth/unlabeled/manifest/",
    )


     pipeline_session = PipelineSession()
    
     TF_FRAMEWORK_VERSION = '2.4.1'

    
    # processing step for feature engineering
    sklearn_processor = FrameworkProcessor(
        estimator_cls=TensorFlow,
        framework_version=TF_FRAMEWORK_VERSION,
        base_job_name = preprocess_job_name,
        command=['python3'],
        py_version="py37",
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        sagemaker_session = pipeline_session
    )

    preprocess_job_name = f"{base_job_prefix}Preprocess"
    
    
    process_output_s3_uri = f's3://{default_bucket}/{base_job_prefix}/{preprocess_job_name}/outputs'#/{uuid.uuid4()}'
    
    step_process_args = script_process.run(
        
        inputs=[
            ProcessingInput(source=input_data,
                                destination="/opt/ml/processing/input/images/"),
            ProcessingInput(source=input_manifest,
                            destination="/opt/ml/processing/input/manifest/"),
        ],
    
    
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--manifest", "manifest",
                       "--images", "images"],
    )
    step_process = ProcessingStep(
        preprocess_job_name,  # choose any name
        step_args = step_process_args,
        cache_config=cache_config    
    )

    # training step for generating model artifacts
    model_path = f"s3://{default_bucket}/{base_job_prefix}/output/models"
    checkpoint_s3_uri = f"s3://{default_bucket}/{base_job_prefix}/output/checkpoints"
    profiler_config = ProfilerConfig(
        system_monitor_interval_millis = 500,
        framework_profile_params = FrameworkProfile(
            detailed_profiling_config = DetailedProfilingConfig(
                start_step = 5, 
                num_steps = 10
            ),
            dataloader_profiling_config = DataloaderProfilingConfig(
                start_step = 7, 
                num_steps = 10
            ),
            python_profiling_config = PythonProfilingConfig(
                start_step = 9, 
                num_steps = 10,
                python_profiler = PythonProfiler.CPROFILE, 
                cprofile_timer = cProfileTimer.TOTAL_TIME
            )
        )
    )
    
    # Set the debugger hook config to save tensors
    debugger_hook_config = DebuggerHookConfig(
        collection_configs = [
            CollectionConfig(name = 'weights'),
            CollectionConfig(name = 'gradients')
        ]
    )

    # Set the rules to analyze tensors emitted during training
    # These specific set of rules will inspect the overall training performance and progress of the model
    rules=[
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ]
    
    hyperparameters = {
        'batch_size': 32,
        'epochs': 10,
        'dropout': 0.76,
        'lr': 0.000019,
        'data_dir': '/opt/ml/input/data'
    }
    
    metric_definitions = [
        {'Name': 'loss', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'acc', 'Regex': 'accuracy: ([0-9\\.]+)'},
        {'Name': 'val_loss', 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'val_acc', 'Regex': 'val_accuracy: ([0-9\\.]+)'}]
    
    if training_instance_count > 1:
        distributions = {
            'mpi': {
                'enabled': True,
                'processes_per_host': 1
            }
        }
        DISTRIBUTION_MODE = 'ShardedByS3Key'
    else:
        distribution = {'parameter_server': {'enabled': False}}
        DISTRIBUTION_MODE = 'FullyReplicated'
                       
    estimator = TensorFlow(entry_point='train_debugger.py',
                           source_dir=os.path.join(BASE_DIR, 'code'),
                           instance_type=training_instance_type,
                           instance_count=training_instance_count,
                           distribution=distribution,
                           hyperparameters=hyperparameters,
                           metric_definitions=metric_definitions,
                           role=role,
                           framework_version=TF_FRAMEWORK_VERSION,
                           py_version='py37',
                           base_job_name=f"{base_job_prefix}-hvd",
                           profiler_config=profiler_config,
                           debugger_hook_config=debugger_hook_config,
                           rules=rules,
                           input_mode='Pipe',
                           script_mode=True,
                           tags=[
                               {
                                   "Key":"TrainingType",
                                   "Value":"OnDemand"
                               }
                           ])

    train_in = TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                             distribution=DISTRIBUTION_MODE)
    val_in   = TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
                             distribution=DISTRIBUTION_MODE)

    inputs = {'train':train_in, 'valid': val_in}

    step_train = TrainingStep(
        name=f"{base_job_prefix}Train",
        estimator=estimator,
        inputs=inputs,
        cache_config=cache_config
    )
    
    evaluation_job_name = f"{base_job_prefix}Evaluation"
    # Processing step for evaluation

    script_eval = FrameworkProcessor(
        estimator_cls=TensorFlow,
        framework_version=TF_FRAMEWORK_VERSION,
        base_job_name = evaluation_job_name,
        command=['python3'],
        py_version="py37",
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        sagemaker_session = pipeline_session)

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, 
                                destination="/opt/ml/processing/input/test"),
                ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["classes"].S3Output.S3Uri, 
                                destination="/opt/ml/processing/input/classes"),
                ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, 
                                destination="/opt/ml/processing/model"),
               ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
         arguments=["--model-file", "model.tar.gz",
                   "--classes-file", "classes.json"],
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name=evaluation_job_name,
        step_args=step_args,
        property_files=[evaluation_report]
        cache_config=cache_config
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name=f"{base_job_prefix}RegisterModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name=f"{base_job_prefix}AccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
