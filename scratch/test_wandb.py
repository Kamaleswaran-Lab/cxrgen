import wandb

run = wandb.init(project = 'test_run', dir = '/hpc/dctrl/ma618/')

config = run.config 
config.learning_rate = 0.001

loss = 1000000
for i in range(10):
    run.log({"loss": loss})
    loss /= 10

run.finish()

