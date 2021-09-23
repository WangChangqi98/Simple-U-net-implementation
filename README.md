# unet-master
Unet network for altrasound image segmentation
## data preparation
structure of project
```
  --project
  	main.py
  	 --altrasound
   		--train
   		--val

all dataset you can access by email：1901684@stu.neu.edu.cn

## training
```
main.py:
		if __name__ == '__main__':
    		batch_size = 4
    		train(batch_size)
```

## testing
```
main.py:
		if __name__ == '__main__':
			ckpt = "weight path"
			test(ckpt)
```

