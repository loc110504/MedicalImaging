# 07 - Full Training Loop Pseudocode

```python
for epoch_num in iterator:
    for i_batch, sampled in enumerate(trainloader):

        image = sampled["image"].cuda()
        scrib = sampled["label"].cuda().long()

        # Forward
        out = model(image, return_all=True)

        logits_s = out["logits_s"]
        logits_l = out["logits_l"]
        logits_g = out["logits_g"]

        probs_s = softmax(logits_s)
        probs_l = softmax(logits_l)
        probs_g = softmax(logits_g)

        # Supervised scribble loss
        loss_scrib = CE(logits_s, scrib, ignore=4) + pDLoss(probs_s, scrib)

        # Teacher heads learn from scribbles too
        loss_aux = 0.5 * (CE(logits_l, scrib, ignore=4) + CE(logits_g, scrib, ignore=4))

        # Student uncertainty
        entropy_s = normalized_entropy(probs_s)
        uncertain_mask = build_uncertain_mask(
            entropy_s, scrib,
            mode=args.uncertain_mode,
            top_ratio=args.uncertain_top_ratio,
            threshold=args.uncertain_threshold,
            ignore_index=4
        )

        # Boundary prior
        boundary = boundary_likelihood(
            image=image,
            probs_s=probs_s,
            lambda_image=args.boundary_lambda_image
        )

        # Region-wise teacher switching
        switch = region_wise_teacher_selection(
            probs_l=probs_l,
            probs_g=probs_g,
            boundary=boundary,
            gamma=args.boundary_gamma
        )

        selected_probs = switch["selected_probs"]
        selected_weight = switch["selected_weight"]
        select_local = switch["select_local"]

        conf_mask = teacher_confidence_mask(selected_probs, args.teacher_conf_threshold)
        pseudo_mask = uncertain_mask & conf_mask

        # Pseudo supervision
        if iter_num >= args.pseudo_warmup:
            loss_pseudo = weighted_soft_ce_loss(
                logits_s,
                selected_probs.detach(),
                pseudo_mask,
                selected_weight.detach()
            )
        else:
            loss_pseudo = logits_s.sum() * 0.0

        # Region-wise HiCo
        if iter_num >= args.pseudo_warmup:
            feat_low_t = select_teacher_feature(out["low_l"], out["low_g"], select_local).detach()
            feat_high_t = select_teacher_feature(out["high_l"], out["high_g"], select_local).detach()

            loss_low = weighted_feature_consistency_loss(
                out["low_s"], feat_low_t, pseudo_mask, selected_weight.detach()
            )
            loss_high = weighted_feature_consistency_loss(
                out["high_s"], feat_high_t, pseudo_mask, selected_weight.detach()
            )
            loss_hico = 0.5 * (loss_low + loss_high)
        else:
            loss_low = logits_s.sum() * 0.0
            loss_high = logits_s.sum() * 0.0
            loss_hico = logits_s.sum() * 0.0

        # Teacher consensus on easy pixels
        easy_mask = (entropy_s < 0.2) & (scrib.unsqueeze(1) == 4)
        loss_consensus = masked_symmetric_kl_loss(probs_l, probs_g, easy_mask)

        # Total
        consistency_weight = sigmoid_rampup(iter_num // 300, args.consistency_rampup)

        loss = (
            loss_scrib
            + args.lambda_aux * loss_aux
            + args.lambda_pseudo * consistency_weight * loss_pseudo
            + args.lambda_hico * consistency_weight * loss_hico
            + args.lambda_consensus * consistency_weight * loss_consensus
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_lr_poly()
        log()
        validate_every_400()
        save_every_3000()
```
