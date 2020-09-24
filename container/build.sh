%%sh

# The name of our algorithm
algorithm_name=neural_network_spam_classifier

#cd container

chmod +x ./source_dir/train.py

#aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.2.0-cpu-py37-ubuntu18.04
#
#docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.2.0-cpu-py37-ubuntu18.04

account=$(aws sts get-caller-identity --query Account --output text)

region=us-east-1

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:version4"

# If the repository doesn't exist in ECR, create it.

aws ecr --region us-east-1 describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr --region us-east-1 create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 540748271236.dkr.ecr.us-east-1.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -f Dockerfile -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
echo ${fullname}
docker push ${fullname}