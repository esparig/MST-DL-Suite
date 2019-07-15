import os.path
import argparse

from PIL import Image

FLAGS = None

def bmp_to_png(data_dir):
    """Create a new folder including all images in png format"""

    if not os.path.exists(data_dir):
        print("[ERROR] The data directory does not exist.")
    else:
        parent, foldername = os.path.split(data_dir)
        print(parent)
        new_folder = os.path.join(parent, "new_"+foldername)
        print(new_folder)
        os.makedirs(new_folder)

    for dirname, dirnames, filenames in os.walk(data_dir):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            print("Folder found: ", os.path.join(dirname, subdirname))
            os.makedirs(os.path.join(new_folder, subdirname))
            print("New folder created: ", os.path.join(new_folder, subdirname))

        # print path to all filenames.
        for filename in filenames:
            if ('.bmp' or '.BMP') in filename:
                img_path = os.path.join(dirname, filename)
                print("BMP file found: ", img_path)
                img = Image.open(img_path)
                new_img_path = os.path.join(new_folder, os.path.basename(dirname), os.path.splitext(filename)[0]+".png")
                new_img = img.save(new_img_path, "png")
                print("PNG image created: ", new_img_path)


    """
    os.makedirs(directory)
    img = Image.open('C:/Python27/image.bmp')
    new_img = img.resize( (256, 256) )
    new_img.save( 'C:/Python27/image.png', 'png')
    """

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      required=True,
      help='Path to folders of labeled images.'
  )
  FLAGS = parser.parse_args()

  print(FLAGS.image_dir)
  bmp_to_png(FLAGS.image_dir)

  

 # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)