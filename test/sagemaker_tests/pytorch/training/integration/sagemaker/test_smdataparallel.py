# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
import os
from sagemaker.pytorch import PyTorch

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT, mnist_path, throughput_path
from ...integration.sagemaker.timeout import timeout
from ...integration.sagemaker.test_distributed_operations import can_run_smmodelparallel, _disable_sm_profiler
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from sagemaker import LocalSession, Session
import boto3


def validate_or_skip_smdataparallel(ecr_image):
    if not can_run_smdataparallel(ecr_image):
        pytest.skip("Data Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smdataparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(
        image_cuda_version.strip("cu")) >= Version("110")


def validate_or_skip_smdataparallel_efa(ecr_image):
    if not can_run_smdataparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on PyTorch 1.8.1 or higher")


def can_run_smdataparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.8.1") and Version(image_cuda_version.strip("cu")) >= Version("110")



def test_smdataparallel_throughput(n_virginia_sagemaker_session, framework_version, n_virginia_ecr_image, instance_types, tmpdir=None):
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_smdataparallel_efa(n_virginia_ecr_image)
        hyperparameters = {
            "size": 64,
            "num_tensors": 20,
            "iterations": 100,
            "warmup": 10,
            "bucket_size": 25,
            "info": "PT-{}-N{}".format(instance_types, 2),
            "nccl": True
        }
        distribution = {'smdistributed': {'dataparallel': {'enabled': True}}}
        pytorch = PyTorch(
            entry_point='smdataparallel_throughput.py',
            role='SageMakerRole',
            instance_count=2,
            instance_type=instance_types,
            source_dir=throughput_path,
            sagemaker_session=n_virginia_sagemaker_session,
            image_uri=n_virginia_ecr_image,
            framework_version=framework_version,
            hyperparameters=hyperparameters,
            distribution=distribution
        )
        pytorch.fit()

n_virginia_region="us-east-1"
n_virginia_sagemaker_session= Session(boto_session=boto3.Session(region_name=n_virginia_region))
framework_version="1.8.1"
n_virginia_ecr_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04"
instance_types="ml.p3dn.24xlarge"
test_smdataparallel_throughput(n_virginia_sagemaker_session, framework_version, n_virginia_ecr_image, instance_types, tmpdir=None)