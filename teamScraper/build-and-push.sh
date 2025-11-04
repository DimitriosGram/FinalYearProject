#!/bin/bash
set -e

error_capture() {
    echo "❌ ERROR: Something went wrong..."
    exit 1
}

trap 'error_capture' ERR

account_id='282423009867'
region='eu-west-2'
repo_name='fyp/teamscrapper'
image_tag='latest'
lambda_function_name='teamScrapper'

echo "🧾 AWS Account ID: ${account_id}"
echo "🌍 Region: ${region}"
echo "📦 ECR Repo: ${repo_name}"
echo "🏷️ Image tag: ${image_tag}"

echo "🔐 Logging in to AWS ECR..."
aws ecr get-login-password --region ${region} | \
    docker login --username AWS --password-stdin "${account_id}.dkr.ecr.${region}.amazonaws.com"

echo "🔍 Checking if ECR repo '${repo_name}' exists..."
aws ecr describe-repositories --repository-name ${repo_name}

image_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/${repo_name}:${image_tag}"

echo "🐳 Building Docker image: ${image_uri}"
DOCKER_BUILDKIT=0 docker build -t "${image_uri}" .


echo "📤 Pushing image to ECR..."
docker push "${image_uri}"
echo "✅ Image pushed successfully."

echo "🚀 Updating Lambda function '${lambda_function_name}' with image"
aws lambda update-function-code \
  --function-name "${lambda_function_name}" \
  --image-uri "${image_uri}"

echo "🎉 Deployment complete: Lambda is now using image ${image_uri}"
