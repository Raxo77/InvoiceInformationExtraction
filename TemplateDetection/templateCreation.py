""""
Template creation in the form of extraction algorithms with algorithms distinguishing
between fixed-value, fixed-position fields; variant-value, fixed-position fields; and
variant-value, variant-position fields.

Proces for the template creation:
Given NER output, i.e., a dict of each entity with the corresponding words, the words are matched
with their corresponding bounding box coordinates. Conceptually, if an invoice template is
then detected and matched to a pre-existing one, the template contains the coordinates of the
words to extract. Practically, this (could/) will be done by cropping the image at the appropriate
location and OCRing the respective crop. For variable-length fields a +/- threshold will be
applied before cropping the image. Practically, the rules (could/) will be encoded as the
coordinates of the words to extract in a JSON (?).
I.e.,
"""



