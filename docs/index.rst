Fast Hough Transform (FHT) project
==================================

Project made for IITP Python course.

Installation
------------

To install Fast Hough Transform project,
run this commands in your terminal:

.. code-block:: console

   $ git clone git@github.com:chousouu/iitp-python.git


Usage
-----

FHT's usage looks like:

.. code-block:: python

   from fasthough.draw_line import hough_transform, read_image

   img_new = read_image(image_path)

   lines_img = hough_transform(img_new)
