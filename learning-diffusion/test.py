        # 假设你需要定义 noise_level，如果没有定义，这里给一个默认值
        noise_level = getattr(self, "noise_level", 1.0) 
        sigma_max = 1.0 # Flow matching 通常 sigma 从 1 到 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                
                # 保持原有的 timestep 处理
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # --- 1. 获取 Model Output (noise_pred) ---
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                # --- 2. CFG 处理 ---
                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # --- 3. 集成自定义 SDE 噪声逻辑 ---
                
                # 获取当前的 sigma (假设 scheduler.sigmas 与 timesteps 对齐)
                # 注意：Flow Matching 中 t 通常直接对应 sigma，或者通过 scheduler 获取
                sigma = self.scheduler.sigmas[i] if hasattr(self.scheduler, "sigmas") else t / 1000.0
                
                # 获取 dt (下一步的 sigma - 当前 sigma)
                if i + 1 < len(self.scheduler.sigmas):
                    next_sigma = self.scheduler.sigmas[i + 1]
                else:
                    next_sigma = 0.0 # 最后一步通常为0
                
                dt = next_sigma - sigma # 注意：这是一个负数，因为 sigma 是从大到小变化的

                # 确保数据类型和设备一致
                sigma = torch.tensor(sigma).to(latents.device, dtype=latents.dtype)
                dt = torch.tensor(dt).to(latents.device, dtype=latents.dtype)

                # [参考代码逻辑 1] 计算 std_dev_t
                # 避免除以0，确保数值稳定性
                denom = 1 - torch.where(sigma == 1, torch.tensor(sigma_max).to(sigma), sigma)
                # 防止 denom 过小
                denom = torch.clamp(denom, min=1e-5) 
                
                std_dev_t = torch.sqrt(sigma / denom) * noise_level

                # [参考代码逻辑 2] 计算 prev_sample_mean (Euler step with noise drift correction)
                # 映射变量: sample -> latents, model_output -> noise_pred
                prev_sample_mean = latents * (1 + std_dev_t**2 / (2 * sigma) * dt) + \
                                   noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt

                # [参考代码逻辑 3] 添加噪声并更新 latents
                variance_noise = torch.randn_like(latents)
                
                # 更新 latents
                # 注意 sqrt(-1 * dt) 因为 dt 是负数
                latents = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

                # 更新进度条 (原有逻辑)
                # progress_bar.update() # 如果是在 callback 中可能需要手动 update
