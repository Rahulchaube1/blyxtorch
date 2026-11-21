# BlyxTorch projects

## Fit a desk sensor

Open `sensor-calibration.blyx`. Five synthetic reference readings determine a gain and an offset. In each epoch two forward-mode derivative evaluations give the gradients for those two parameters. The program reports 18.625 and 22.375 degrees for two readings omitted from training. Replace either final input and run again.

## Predict print-queue waits

The two features are pages divided by 100 and waiting jobs divided by 10. The binary label marks a short or long wait in a synthetic print queue. Training uses 16 rows. Six different labelled rows are reserved for evaluation. The four unlabelled rows in `print-queue/new-jobs.csv` are for prediction. These fixtures teach the workflow; decisions for a real printer need measured local data.

Open `queue-classifier.blyx` to use the saved weights. It prints short, long, short and long waits. The model has two inputs, four tanh hidden units and one output. The checkpoint is reproduced by the following command with seed 7. From the projects folder, with BlyxTorch 0.2.0 on PATH, run:

~~~sh
blyxtorch train print-queue/train.csv 180 0.03 4 adam 7
blyxtorch train-score print-queue/train.csv print-queue/holdout.csv 180 0.03 4 adam 7
blyxtorch train-predict print-queue/train.csv print-queue/new-jobs.csv 180 0.03 4 adam 7
~~~

## Use the editor

Install the BlyxTorch extension. Open `BlyxTorch.code-workspace` from this folder. In a trusted workspace choose **Terminal → Run Task** and select **BlyxTorch: Evaluate print queue** or **BlyxTorch: Predict new print jobs**. Both tasks run the native trainer bundled with the extension. You can edit the CSV files and rerun them. For a portable project, open either `.blyx` file and choose **Blyx: Run File** from the command palette. The two files also run with `blyxc run`. [Project guide](https://www.blyx-lang.space/packages/blyxtorch#projects)

## Connect with Rahul Chaube

[Portfolio](https://www.rahulchaube1.com/) · [GitHub](https://github.com/Rahulchaube1) · [LinkedIn](https://www.linkedin.com/in/rahulchaube1) · [Hugging Face](https://huggingface.co/oncody) · [X](https://x.com/RahulChaube_) · [EverestQ](https://www.everestqai.com/) · [Contact](https://www.rahulchaube1.com/contact)

[Blyx website](https://www.blyx-lang.space/) · [BlyxTorch](https://www.blyx-lang.space/packages/blyxtorch) · [Handbook](https://doc.blyx-lang.space/) · [Community](https://www.blyx-lang.space/community)

Support Rahul’s open-source work through [GitHub Sponsors](https://github.com/sponsors/Rahulchaube1) or [Buy Me a Coffee](https://buymeacoffee.com/rahulchaube).
