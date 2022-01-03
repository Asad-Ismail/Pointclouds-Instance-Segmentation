
import sagemaker
sess=sagemaker.Session(default_bucket="sagemaker-us-east-1-470086202700")
role= sagemaker.get_execution_role()
account =sess.boto_session.client('sts').get_caller_identity()['Account']
region= sess.boto_session.region_name

repo_name="pepp-detectron2"
base_job_name="pepp-detectron2"
train_input_path="s3://sagemaker-us-east-1-470086202700/pepper-segmentation-s-dataset/training"
validation_input_path="s3://sagemaker-us-east-1-470086202700/pepper-segmentation-s-dataset/validation"

image_name  = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, repo_name)
output_path="s3://sagemaker-us-east-1-470086202700/pepper-segmentation-s-dataset/output"

train_instance_type ="ml.p3.8xlarge"

estimator = sagemaker.estimator.Estimator(
                       image_uri=image_name,
                       base_job_name=base_job_name,
                       role=role, 
                       instance_count=1, 
                       instance_type=train_instance_type,
                       output_path=output_path,
                       sagemaker_session=sess,
                       hyperparameters={
                              'epochs': 20,
                              'batch-size': 130,
                              'lr': 0.01}
                        )

estimator.fit({'training': train_input_path, 'validation': validation_input_path})

