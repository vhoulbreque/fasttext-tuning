# fasttext-tuning

How to find the best parameters to tune the model? Let's use genetic algorithms!

To make it work, don't forget to
```bash
$ cd src
$ chmod +x fasttext-tuning
```

And add this line to your `~/.bashrc`:
```
export PATH="project_path/src:$PATH"
```

and execute `source ~/.bashrc`.

Then, install the python libraries with:

```
$ pip install -r requirements.txt
```

## TODO:

 - [X] get_metrics is dependent on the labels names
 - [X] no time output
 - [ ] better print output
 - [X] save best model
 - [X] gestion of train/test files to be enhanced
 - [ ] possibility to choose accuracy, recall, f1 score
