# NOTE: If you change these variables then you also have to
# change them in the model and retrain it
num_classes = 17

img_dim = (128, 128, 3)
thresholds = [0.2] * num_classes


#These are variables which can be change d according to the preference
ymap = {0: 'agriculture',
 1: 'artisinal_mine',
 2: 'bare_ground',
 3: 'blooming',
 4: 'blow_down',
 5: 'clear',
 6: 'cloudy',
 7: 'conventional_mine',
 8: 'cultivation',
 9: 'habitation',
 10: 'haze',
 11: 'partly_cloudy',
 12: 'primary',
 13: 'road',
 14: 'selective_logging',
 15: 'slash_burn',
 16: 'water'}

UPLOAD_IMAGES_PATH = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
