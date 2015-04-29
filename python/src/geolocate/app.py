##
#  Copyright (c) 2015, Derek Ruths, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##
import argparse
import json
import simplejson
import jsonlib
import logging
import os, os.path
import datetime
import gzip
import time

from collections import defaultdict
from gimethod import gimethod_subclasses, GIMethod
from dataset import Dataset, posts2dataset
from sparse_dataset import SparseDataset

logger = logging.getLogger(__name__)

def get_method_by_name(name):
	# get the geoinference class
	candidates = filter(lambda x: x.__name__ == name, gimethod_subclasses())

	if len(candidates) == 0:
		logger.fatal('No geoinference named "%s" was found.' % name)
		logger.info('Available methods are: %s' % ','.join([x.__name__ for x in gimethod_subclasses()]))
		quit()

	if len(candidates) > 1:
		logger.fatal('More than one geoinference method named "%s" was found.')
		quit()

	return candidates[0]

def ls_methods(args):
	"""
	Print out the set of methods that the tool knows about.
	"""
	for x in gimethod_subclasses():
		print '\t' + x.__name__ 

def create_folds(args): 
	parser = argparse.ArgumentParser(prog='geoinf create_folds', description='creates a set of data partitions for evaluating with cross-fold validation')
	parser.add_argument('-f', '--force', help='overwrite the output model directory if it already exists')
	parser.add_argument('dataset_dir', help='a directory containing a geoinference dataset')
	parser.add_argument('num_folds', help='the number of folds into which the dataset should be divided')
	parser.add_argument('fold_dir', help='a (non-existent) directory that will contain the information on the cross-validation folds')

	args = parser.parse_args(args)

	# Confirm that the output directory doesn't exist
	if not os.path.exists(args.fold_dir): #and not args.force:
		#raise Exception, 'output fold_dir cannot already exist'
                os.mkdir(args.fold_dir)

	# Decide on the number of folds
	num_folds = int(args.num_folds)
	if num_folds <= 1:
		raise Exception, 'The number of folds must be at least two'

        # Initialize the output streams.  Rather than keeping things in memory,
        # we batch the gold standard posts by users (one at a time) and then
        # stream the user's gold standard posts (if any) to the output streams
        output_held_out_post_ids_file_handles = []
        output_held_out_user_ids_file_handles = []
        output_gold_loc_file_handles = []
        output_posts_file_handles = []
	cf_info_fh = open(os.path.join(args.fold_dir, "folds.info.tsv"), 'w')

	for i in range(0, num_folds):
		fold_name = "fold_%d" % i
                # All the IDs of the gold posts in this fold are written here
		fold_posts_ids_fh = open(os.path.join(args.fold_dir, fold_name + ".post-ids.txt"), 'w')
                output_held_out_post_ids_file_handles.append(fold_posts_ids_fh)

                # All the IDs of the users with gold posts are written here
		fold_users_ids_fh = open(os.path.join(args.fold_dir, fold_name + ".user-ids.txt"), 'w')
                output_held_out_user_ids_file_handles.append(fold_users_ids_fh)

                # All the lat/lon and IDs of the gold posts are written here
		gold_loc_fh = open(os.path.join(args.fold_dir, fold_name + ".gold-locations.tsv"), 'w')
                output_gold_loc_file_handles.append(gold_loc_fh)

                # The users.json.gz file with the gold data (used for testing)
		gold_loc_fh = gzip.open(os.path.join(args.fold_dir, fold_name + ".users.json.gz"), 'w')
                output_posts_file_handles.append(gold_loc_fh)
                cf_info_fh.write("%s\t%s.post-ids.txt\t%s.user-ids.txt\t%s.users.json.gz\n" 
                                 % (fold_name, fold_name, fold_name, fold_name))
        cf_info_fh.close()

	# Load the dataset
	ds = SparseDataset(args.dataset_dir)

	logger.debug('Extracting gold-standard posts')
	num_users = 0
        num_posts = 0
        num_gold_users = 0
        num_gold_posts = 0

	# Iterate over the dataset looking for posts with geo IDs that we can
	# use as a gold standard
	for user in ds.user_iter():
                gold_posts = []
                gold_post_id_to_loc = {}
                user_id = user['user_id']
                num_posts += len(user['posts'])
                for post in user['posts']:
                        if "geo" in post:
                                post_id = post['id']
                                loc = post['geo']['coordinates']
                                gold_post_id_to_loc[post_id] = loc
                                gold_posts.append(post)
                # If this user had any gold locations, add them as folds
                if len(gold_posts) > 0:
                        num_gold_posts += len(gold_posts)
                        fold_to_use = num_gold_users % num_folds
                        num_gold_users += 1
                        
                        output_held_out_user_ids_file_handles[fold_to_use].write("%s\n" % user['user_id'])

                        for post_id, loc in gold_post_id_to_loc.iteritems():
                                output_held_out_post_ids_file_handles[fold_to_use].write("%d\n" % post_id)
                                output_gold_loc_file_handles[fold_to_use].write("%d\t%s\t%f\t%f\n" % (post_id, user_id, loc[0], loc[1]))
                        # Lazily mutate the existing user object and the dump
                        # that object to the fold's user.json.gz 
                        user['posts'] = gold_posts
                        output_posts_file_handles[fold_to_use].write("%s\n" % simplejson.dumps(user))
                        
                num_users += 1
		if num_users % 100000 == 0:
			logger.debug('Processed %d users, saw %d gold so far (%d posts of %d (%f))' 
                                     % (num_users, num_gold_users, num_gold_posts, num_posts,
                                        float(num_gold_posts) / num_posts))

        for fh in output_posts_file_handles:
                fh.close()
        for fh in output_held_out_post_ids_file_handles:
                fh.close()
        for fh in output_held_out_user_ids_file_handles:
                fh.close()
        for fh in output_gold_loc_file_handles:
                fh.close()

	logger.debug('Saw %d gold standard users in %d total' % (num_gold_users, num_users))

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
	yield l[i:i+n]

def cross_validate(args): 
	parser = argparse.ArgumentParser(prog='geoinf cross_validate', description='evaluate a geocinference method using cross-validation')
	parser.add_argument('-f', '--force', help='overwrite the output model directory if it already exists')
	parser.add_argument('method_name', help='the method to use')
	parser.add_argument('method_settings', help='a json file containing method-specific configurations')
	parser.add_argument('dataset_dir', help='a directory containing a geoinference dataset')
	parser.add_argument('fold_dir', help='the name of the directory containing information on the cross-validation folds')
	parser.add_argument('results_dir', help='a (non-existent) directory where the evaluation results will be stored')
        parser.add_argument('--fold', nargs=1, 
                            help='runs just that fold from the cross-fold dataset')
        parser.add_argument('--location-source', nargs=1, 
                            help='specifies the source of ground-truth locations')

	args = parser.parse_args(args)

	# confirm that the output directory doesn't exist
#	if os.path.exists(args.results_dir) and not args.force:
#		raise Exception, 'output results_dir cannot already exist'

	if not os.path.exists(args.results_dir): #and not args.force:
		#raise Exception, 'output fold_dir cannot already exist'
                os.mkdir(args.results_dir)


	# load the method
	method = get_method_by_name(args.method_name)

	# load the data
	with open(args.method_settings, 'r') as fh:
		settings = json.load(fh)

        specific_fold_to_run = args.fold
        if specific_fold_to_run:
                specific_fold_to_run = specific_fold_to_run[0]
        location_source = args.location_source
        if location_source:
                logger.debug('Using %s as the source of ground truth location' % location_source)
                location_source = location_source[0]
                settings['location_source'] = location_source

                
        print "running fold %s" % (specific_fold_to_run)

	# Load the folds to be used in the dataset
	cfv_fh = open(os.path.join(args.fold_dir, 'folds.info.tsv'))

	# Each line contains two files specifying the post IDs to be held out
	# from the full dataset (for that fold) and the corresponding file in
	# the fold_dir containing the testing data for that fold
	for line in cfv_fh:
                line = line.strip()
		fold_name, testing_post_ids_file, testing_user_ids_file, testing_users_file = line.split("\t")

                # Skip this fold if the user has told us to run only one fold by name
                if specific_fold_to_run is not None and fold_name != specific_fold_to_run:
                        continue
		
                logger.debug('starting processing of fold %s' % fold_name)
                
		# Read in the post IDs to exclude
		testing_post_ids = set()
		tpi_fh = open(os.path.join(args.fold_dir, testing_post_ids_file.replace('held-out-','')))
		for id_str in tpi_fh:
			testing_post_ids.add(id_str.strip())
		tpi_fh.close()

		# Read in the user IDs to exclude
		testing_user_ids = set()
		tpi_fh = open(os.path.join(args.fold_dir, testing_user_ids_file.replace('held-out-','')))
		for id_str in tpi_fh:
			testing_user_ids.add(id_str.strip())
		tpi_fh.close()

                logger.debug('Loaded %d users whose location data will be held out' % len(testing_user_ids))

		# load the dataset
                training_data = None
                if not location_source is None:
                        training_data = SparseDataset(args.dataset_dir, excluded_users=testing_user_ids, default_location_source=location_source)
                else:
                        training_data = SparseDataset(args.dataset_dir, excluded_users=testing_user_ids)
                
		# load the method
		method = get_method_by_name(args.method_name)
		method_inst = method()
                
                # Create the temporary directory that will hold the model for
                # this fold
                model_dir = os.path.join(args.results_dir, fold_name)
                if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                
		# Train on the datset, holding out the testing post IDs
		model = method_inst.train_model(settings, training_data, None)

                logger.debug('Finished training during fold %s; beginning testing' % fold_name)

                logger.debug("Reading testing data from %s" % (os.path.join(args.fold_dir,testing_users_file)))

		testing_data = Dataset(args.fold_dir, users_file=os.path.join(args.fold_dir,testing_users_file))

                logger.debug("Writing results to %s" % (os.path.join(args.results_dir, fold_name + ".results.tsv.gz")))
                
		out_fh = gzip.open(os.path.join(args.results_dir, fold_name + ".results.tsv.gz"), 'w')

                num_tested_users = 0
                num_tested_posts = 0
                seen_ids = set()
		for user in testing_data.user_iter():
			user_id = user['user_id']
			posts = user['posts']

			locs = model.infer_posts_locations_by_user(user_id, posts)

			if len(locs) != len(posts):
                                print "#WUT %d != %d" % (len(locs), len(posts))
                        
                        num_located_posts = 0 
                        num_tested_posts += len(posts)
			for loc, post in zip(locs, posts):
                                pid = post['id']
                                if pid in seen_ids:
                                        continue
                                seen_ids.add(pid)
                                if not loc is None:
                                        out_fh.write('%s\t%f\t%f\n' % (post['id'], loc[0], loc[1]))
                                        num_located_posts += 1
                        num_tested_users += 1
                        if num_tested_users % 10000 == 0:
                                logger.debug('During testing of fold %s, processed %d users, %d posts, %d located' % (fold_name, num_tested_users, num_tested_posts, num_located_posts))

		out_fh.close()
                logger.debug('Finished testing of fold %s' % fold_name)



def train(args):
	parser = argparse.ArgumentParser(prog='geoinf train',description='train a geoinference method on a specific dataset')
	parser.add_argument('-f','--force',help='overwrite the output model directory if it already exists')
	parser.add_argument('method_name',help='the method to use')
	parser.add_argument('method_settings',help='a json file containing method-specific configurations')
	parser.add_argument('dataset_dir',help='a directory containing a geoinference dataset')
	parser.add_argument('model_dir',help='a (non-existing) directory where the trained model will be stored')
        parser.add_argument('--location-source', nargs=1, 
                            help='specifies the source of ground-truth locations')
		
	args = parser.parse_args(args)

	# confirm that the output directory doesn't exist
	if os.path.exists(args.model_dir) and not args.force:
		raise Exception, 'output model_dir cannot exist'

	# load the method
	method = get_method_by_name(args.method_name)

	# load the data
	with open(args.method_settings,'r') as fh:
		settings = json.load(fh)

        location_source = args.location_source
        if location_source:
                location_source = location_source[0]
                logger.debug('Using %s as the source of ground truth location'
                             % location_source)
                settings['location_source'] = location_source



	# load the dataset
	ds = None #Dataset(args.dataset_dir)
        if not location_source is None:
                ds = SparseDataset(args.dataset_dir, default_location_source=location_source)
        else:
                ds = SparseDataset(args.dataset_dir)


	# load the method
	method = get_method_by_name(args.method_name)
	method_inst = method()

        start_time = time.time()       
	method_inst.train_model(settings,ds,args.model_dir)
        end_time = time.time()
	logger.info('Trained model %s on dataset %s in %f seconds' 
                    % (args.method_name, args.dataset_dir, end_time - start_time))

	# drop some metadata into the run method
	# run the method
	# gi_inst = method()
	# gi_inst.train(settings,ds,args.model_dir)

	return

def infer(args,by_user=False):
	prog_name = 'geoinf'
	if by_user:
		description='infer the location of posts in a dataset using a specific inference method. Posts will be provided to the method grouped by user.'
		prog_name += ' infer_by_user'
	else:
		description='infer the location of posts in a dataset using a specific inference method. Posts will be provided to the method one at a time.'
		prog_name += ' infer_by_post'

	parser = argparse.ArgumentParser(prog=prog_name,description=description)
	parser.add_argument('-f','--force',action='store_true',help='overwrite the output file if it already exists')
	parser.add_argument('-s','--settings',help='a json file of settings to be passed to the model',nargs=1)
	parser.add_argument('method_name',help='the type of method to use for inference')
	parser.add_argument('model_dir',help='the directory of a model that was constructed using the train procedure')
	parser.add_argument('dataset',help='a json specification for the dataset to infer locations on')
	parser.add_argument('infer_file',help='the file that the inferences will be written to')
		
	logger.debug('infer args = %s' % str(args))
	args = parser.parse_args(args)

	# load the infer settings if necessary
	settings = {}
	if args.settings:
		with open(args.settings,'r') as fh:
			settings = json.load(fh)
	
	if os.path.exists(args.infer_file) and not args.force:
		raise Exception, 'output infer_file cannot exist'

	# load the method
	method = get_method_by_name(args.method_name)
	method_inst = method()
	model = method_inst.load_model(args.model_dir,settings)

	# load the dataset
	ds = SparseDataset(args.dataset)

	# get the output file ready
	outfh = open(args.infer_file,'w')

	# write settings to the first line
	outfh.write('%s\n' % json.dumps({'method': args.method_name, 
									 'settings': settings, 
									 'dataset': args.dataset,
									 'by_user': by_user}))
	
	# locate all the posts
	logger.info('inferring locations for posts')	
	if by_user:
                num_posts_seen = 0
                num_posts_located = 0
                num_users_seen = 0
		for user in ds.user_iter():
			user_id = user['user_id']
			posts = user['posts']

			locs = model.infer_posts_locations_by_user(user_id,posts)

			assert len(locs) == len(posts)
                        num_users_seen += 1

			for loc,post in zip(locs,posts):
                                num_posts_seen += 1
				if not loc is None:
                                        num_posts_located += 1
					outfh.write('%s\t%f\t%f\n' % (post['id'],loc[0],loc[1]))

                                if num_posts_seen % 10000 == 0:
                                        logger.debug("Saw %d users, %d posts, %d of which were located" % (num_users_seen, num_posts_seen, num_posts_located))
	else:
                num_posts_seen = 0
                num_posts_located = 0
		for post in ds.post_iter():
                        user_id = post['user']['id_str']
			loc = model.infer_post_location(post)
                        num_posts_seen += 1
			if not loc is None:
                                outfh.write('%s\t%f\t%f\n' % (post['id'],loc[0],loc[1]))
                                num_posts_located += 1
                        if num_posts_seen % 10000 == 0:
                                logger.debug("Saw %d posts, %d of which were located" % (num_posts_seen, num_posts_located))

	outfh.close()

	# done

def build_dataset(args):
	parser = argparse.ArgumentParser(prog='geoinf build_dataset',description='build a new dataset')
	parser.add_argument('-f','--force',action='store_true')
	parser.add_argument('dataset_dir',help='the directory to put the dataset in')
	parser.add_argument('posts_file',help='the posts.json.gz file to use')
	parser.add_argument('user_id_field',help='the field name holding the user id of the post author')
	parser.add_argument('mention_field',help='the field name holding the list of user ids mentioned in a post')

	args = parser.parse_args(args)

#	uid_field_name = args.user_id_field
	uid_field_name = args.user_id_field.split('.')[::-1]
	mention_field_name = args.mention_field.split('.')[::-1]
	posts2dataset(args.dataset_dir,args.posts_file,
				  lambda x: (lambda a: lambda dic, ind: a(a, dic, ind))(lambda s, dic, ind: str(dic) if ind == -1  else s(s,dic[uid_field_name[ind]], ind-1))(x, len(uid_field_name)-1),
				  lambda x: (lambda a: lambda dic, ind: a(a, dic, ind))(lambda s, dic, ind: str(dic) if ind == -1 else (s(s,dic.get(mention_field_name[ind],[]), ind-1) if type(dic) == dict else map(lambda d: s(s,d,ind), dic)))(x, len(mention_field_name)-1),
				  force=args.force)
	
	# done

def main():
	parser = argparse.ArgumentParser(prog='geoinf',description='run a geolocation inference method on a dataset')
	parser.add_argument('-l','--log_level',
						choices=['DEBUG','INFO','WARN','ERROR','FATAL'],
						default='INFO',help='set the logging level')
	parser.add_argument('action',choices=['train','infer_by_post','infer_by_user',
					      'ls_methods','build_dataset','create_folds','cross_validate'],
			help='indicate whether to train a new model or infer locations')
	parser.add_argument('action_args',nargs=argparse.REMAINDER,
			help='arguments specific to the chosen action')

	args = parser.parse_args()

	logging.basicConfig(level=eval('logging.%s' % args.log_level),
						format='%(message)s')

	try:
		if args.action == 'train':
			train(args.action_args)
		elif args.action == 'ls_methods':
			ls_methods(args.action_args)
		elif args.action == 'infer_by_post':
			infer(args.action_args,False)
		elif args.action == 'infer_by_user':
			infer(args.action_args,True)
		elif args.action == 'build_dataset':
			build_dataset(args.action_args)
		elif args.action == 'create_folds':
			create_folds(args.action_args)
		elif args.action == 'cross_validate':
			cross_validate(args.action_args)

		else:
			raise Exception, 'unknown action: %s' % args.action

	except Exception, e:
		logger.exception(e)	

	# done!

if __name__ == '__main__':
	main()


