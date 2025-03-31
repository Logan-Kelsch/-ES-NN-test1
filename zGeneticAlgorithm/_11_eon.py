import _10_era as _10
import _01_initialization as _1
import _02_evaluation as _2

def autobatch_eon(
	eon_num		:	int,
	population	:	any,
	criteria	:	str,
	filter_tightening	:	dict|None,
	era_kwargs	:	dict,
	spec_batch_size	:	int	=	100
):

	era_kwargs['with_array'] = True
	era_survivors = []

	recomp_loaded_batches = _1.make_population_batches(
    	population=population,
		batch_size=spec_batch_size
	)

	num_eras = len(recomp_loaded_batches)

	for era in range(num_eras):

		endera = _10.era(
			eon_num=eon_num,
			era_num=era,
			new_population= recomp_loaded_batches[era],
			**era_kwargs
		)

		era_survivors.append(endera)

	#filter tightening option
	if(filter_tightening != {}):
		era_kwargs['strict_filter_kwargs'][filter_tightening.keys()[0]] = filter_tightening.values()[0]

	eon_members = _1.combine_populations(era_survivors)

	print(f"Completed batched eras...")

	#era_kwargs['with_array'] = True

	endeon = _10.era(
		eon_num=eon_num+1,
		era_num=0,
		new_population= eon_members,
		**era_kwargs
	)

	return endeon