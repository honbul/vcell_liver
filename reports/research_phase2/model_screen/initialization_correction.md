# LoRA initialization seed correction

M03_full4096_fold0_seed42_v1 seeded the classifier head and data order, but called PEFT adapter initialization before torch.manual_seed. Its first adapter A initialization was therefore not controlled by the declared seed. The gradient/reload integration probe was separately seeded and remains valid.

The v1 efficacy run is preserved as an initialization-uncontrolled run. It is excluded from confirmation candidate selection. The same registered rank8/alpha16, LR and data condition is rerun as M03_full4096_fold0_seed42_v2 after seeding before both encoder/adapter construction and head construction. This is a correctness rerun, with no hyperparameter change. The draft model confirmation selection was invalidated before any adapted confirmation job was launched. The corrected selector will use v2 inner-validation scores.

Frozen NN and partial runs initialized the shared head after their seed reset, and use fully loaded frozen pretrained base weights. Their model initialization does not include random trainable adapters. They remain usable; a loading audit checks for missing pretrained parameters.
