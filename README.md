# unet-master

Unet network for altrasound image segmentation

## structure of project
'''
  --project
  	main.py
	unet.py
	dataset.py
  	altrasound.zip

all dataset you can access by email：1901684@stu.neu.edu.cn

*training

edit main.py:

		if __name__ == '__main__':
		
    		batch_size = 4
		
    		train(batch_size)


*testing

edit main.py:

		if __name__ == '__main__':
		
			ckpt = "weight file path"
			
			test(ckpt)


All details are marked with Chinese!
