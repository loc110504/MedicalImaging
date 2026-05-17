# 04 - Full Pseudocode

```python
for epoch_num in iterator:
    for i_batch, sampled in enumerate(trainloader):
        image = sampled['image'].cuda()
        scrib = sampled['label'].cuda().long()

        # Teacher forward
        with torch.no_grad():
            logits_fast, high_fast, low_fast = teacher_fast(image)
            logits_slow, high_slow, low_slow = teacher_slow(image)
            probs_fast = torch.softmax(logits_fast, dim=1)
            probs_slow = torch.softmax(logits_slow, dim=1)

        # Student forward
        logits_stu, high_stu, low_stu = model(image)
        probs_stu = torch.softmax(logits_stu, dim=1)

        # Scribble supervised loss
        loss_ce = ce_loss(logits_stu, scrib)
        loss_dice = dice_loss(probs_stu, scrib.unsqueeze(1))
        loss_scrib = loss_ce + loss_dice

        # Student uncertainty
        entropy_stu = normalized_entropy(probs_stu)
        uncertain_mask = build_uncertain_mask(entropy_stu, scrib, args.uncertain_mode,
                                              args.uncertain_top_ratio, args.uncertain_threshold, 4)

        # Teacher reliability signals
        entropy_fast = normalized_entropy(probs_fast)
        entropy_slow = normalized_entropy(probs_slow)
        disagreement = js_divergence_map(probs_fast, probs_slow)
        boundary = boundary_likelihood(image, probs_stu, args.boundary_lambda_image) \
                   if args.use_boundary_prior else torch.zeros_like(disagreement)

        # Region-wise teacher arbitration
        switch = temporal_teacher_arbitration(probs_fast, probs_slow,
                                              entropy_fast, entropy_slow,
                                              disagreement, boundary,
                                              args.gamma_boundary,
                                              args.gamma_disagree,
                                              args.gamma_stable)
        selected_probs = switch['selected_probs']
        selected_weight = switch['selected_weight']
        select_fast = switch['select_fast']

        # Pseudo-label mask
        conf_mask = teacher_confidence_mask(selected_probs, args.teacher_conf_threshold)
        pseudo_mask = uncertain_mask & conf_mask

        # Pseudo loss
        if iter_num >= args.pseudo_warmup:
            loss_pseudo = weighted_soft_ce_loss(logits_stu, selected_probs.detach(),
                                                pseudo_mask, selected_weight.detach())
        else:
            loss_pseudo = logits_stu.sum() * 0.0

        # Region-wise HiCo
        if iter_num >= args.pseudo_warmup:
            selected_low = select_teacher_feature(low_fast, low_slow, select_fast).detach()
            selected_high = select_teacher_feature(high_fast, high_slow, select_fast).detach()
            loss_low = weighted_feature_consistency_loss(low_stu, selected_low, pseudo_mask, selected_weight.detach())
            loss_high = weighted_feature_consistency_loss(high_stu, selected_high, pseudo_mask, selected_weight.detach())
            loss_hico = 0.5 * (loss_low + loss_high)
        else:
            loss_low = loss_high = loss_hico = logits_stu.sum() * 0.0

        # Stable-region slow teacher loss
        easy_mask = (entropy_stu < args.easy_threshold) & (scrib.unsqueeze(1) == 4)
        if iter_num >= args.pseudo_warmup:
            loss_stable = weighted_soft_ce_loss(logits_stu, probs_slow.detach(), easy_mask, None)
        else:
            loss_stable = logits_stu.sum() * 0.0

        # Total loss
        w = get_current_consistency_weight(iter_num // 300, args)
        loss = loss_scrib + w * args.lambda_pseudo * loss_pseudo \
                          + w * args.lambda_hico * loss_hico \
                          + w * args.lambda_stable * loss_stable

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update both EMA teachers
        update_ema_variables(model, teacher_fast, args.alpha_fast, iter_num)
        update_ema_variables(model, teacher_slow, args.alpha_slow, iter_num)

        # LR, logging, validation, save checkpoint as in train_acdc.py
```
