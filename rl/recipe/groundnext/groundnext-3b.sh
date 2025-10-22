#!/bin/bash
set -x
ulimit -n 65535

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 

/mnt/home/.conda/envs/verl-vllm/bin/python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=/home/GroundCUA/rl/data/GroundCUA/train_data.parquet \
    data.val_files=/home/GroundCUA/rl/data/GroundCUA/val_data.parquet \
    data.train_batch_size=64 \
    data.dataloader_num_workers=8 \
    data.max_prompt_length=10000 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    custom_reward_function.path=/home/GroundCUA/rl/recipe/groundnext/reward_clipped.py \
    custom_reward_function.name=gui_reward_function \
    actor_rollout_ref.model.path=/home/GroundCUA/sft/checkpoints/ \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='GroundCUA' \
    trainer.experiment_name='GroundNext-rl-3b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.log_val_generations=5 \
    trainer.total_epochs=1
