import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams
import ontology
from clean_dataset import clean_slot_values

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


class MultiWozEvaluator(object):
    def __init__(self, reader, cfg):
        self.reader = reader
        self.domains = ontology.all_domains
        self.domain_files = self.reader.domain_files
        self.all_data = self.reader.data
        self.test_data = self.reader.test
        self.cfg = cfg
        self.set_attribute()

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for d, s_list in ontology.informable_slots.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        self.data_prefix = self.cfg.data_prefix
        self.mapping_pair_path = self.data_prefix + '/multi-woz/mapping.pair'

    def wrap_evaluation_result(self, result_list):
        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                         'dspn_gen', 'dspn', 'bspn', 'pointer']
        
        id_to_list_dict = {}
        id_list, visited_set = [], set()
        for item in result_list:
            one_dial_id = item['dial_id']
            try:
                id_to_list_dict[one_dial_id].append(item)
            except KeyError:
                id_to_list_dict[one_dial_id] = [item]
            if one_dial_id in visited_set:
                pass
            else:
                id_list.append(one_dial_id)
                visited_set.add(one_dial_id)
        #print (len(id_list))
        res_list = []
        for one_id in id_list:
            one_session_list = id_to_list_dict[one_id]
            session_len = len(id_to_list_dict[one_id])
            entry = {'dial_id': one_id, 'trun_num': session_len}
            for f in field[2:]:
                entry[f] = '' # ???
            res_list.append(entry)
            for one_turn in one_session_list:
                res_list.append(one_turn)
        return res_list

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials


    def validation_metric(self, data):
        data = self.wrap_evaluation_result(data)
        bleu = self.bleu_metric(data)
        # accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(data,
                                                                                        same_eval_as_cambridge=self.cfg.same_eval_as_cambridge)
        return bleu, success, match

    def set_attribute(self):
        self.cfg.use_true_prev_bspn = True
        self.cfg.use_true_prev_aspn = True
        self.cfg.use_true_db_pointer = True
        self.cfg.use_true_prev_resp = True
        self.cfg.use_true_pv_resp = True
        self.cfg.same_eval_as_cambridge = True
        self.cfg.use_true_domain_for_ctr_eval = True
        self.cfg.use_true_bspn_for_ctr_eval = False
        self.cfg.use_true_curr_bspn = False
        self.cfg.use_true_curr_aspn = False

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [],[]
        for row in data:
            if eval_dial_list and row['dial_id'] +'.json' not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc

    def domain_eval(self, data, eval_dial_list = None):
        dials = self.pack_dial(data)
        corr_single, total_single, corr_multi, total_multi = 0, 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_pred = []

            prev_constraint_dict = {}
            prev_turn_domain = ['general']

            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                true_domains = self.reader.dspan_to_domain(turn['dspn'])
                if self.cfg.enable_dspn:
                    pred_domains = self.reader.dspan_to_domain(turn['dspn_gen'])
                else:
                    turn_dom_bs = []
                    if self.cfg.enable_bspn and not self.cfg.use_true_bspn_for_ctr_eval and \
                        (cfg.bspn_mode == 'bspn' or self.cfg.enable_dst):
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
                    else:
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn'])
                    for domain in constraint_dict:
                        if domain not in prev_constraint_dict:
                            turn_dom_bs.append(domain)
                        elif prev_constraint_dict[domain] != constraint_dict[domain]:
                            turn_dom_bs.append(domain)
                    aspn = 'aspn' if not self.cfg.enable_aspn else 'aspn_gen'
                    turn_dom_da = []
                    for a in turn[aspn].split():
                        if a[1:-1] in ontology.all_domains + ['general']:
                            turn_dom_da.append(a[1:-1])

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]
                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    turn['dspn_gen'] = ' '.join(['['+d+']' for d in turn_domain])
                    pred_domains = {}
                    for d in turn_domain:
                        pred_domains['['+d+']'] = 1

                if len(true_domains) == 1:
                    total_single += 1
                    if pred_domains == true_domains:
                        corr_single += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'
                else:
                    total_multi += 1
                    if pred_domains == true_domains:
                        corr_multi += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'

            # dialog inform metric record
            dial[0]['wrong_domain'] = ' '.join(wrong_pred)
        accu_single = corr_single / (total_single + 1e-10)
        accu_multi = corr_multi / (total_multi + 1e-10)
        return accu_single * 100, accu_multi * 100, total_multi


    def context_to_response_eval(self, data, eval_dial_list = None, same_eval_as_cambridge=False):
        dials = self.pack_dial(data)
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}
            if '.json' not in dial_id and '.json' in list(self.all_data.keys())[0]:
                dial_id = dial_id + '.json'
            for domain in ontology.all_domains:
                if self.all_data[dial_id]['goal'].get(domain):
                    true_goal = self.all_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(dial, goal, reqs, counts,
                                                                    same_eval_as_cambridge=same_eval_as_cambridge)

            successes += success
            matches += match
            dial_num += 1

        succ_rate = successes/( float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100
        return succ_rate, match_rate, counts, dial_num


    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                                          soft_acc=False, same_eval_as_cambridge=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
         #'id'
        requestables = self.requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0: 
                continue 
            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if same_eval_as_cambridge:
                        # [restaurant_name], [hotel_name] instead of [value_name]
                        if self.cfg.use_true_domain_for_ctr_eval:
                            dom_pred = [d[1:-1] for d in turn['dspn'].split()]
                        else:
                            dom_pred = [d[1:-1] for d in turn['dspn_gen'].split()]
                        # else:
                        #     raise NotImplementedError('Just use true domain label')
                        if domain not in dom_pred:  # fail
                            continue
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if not self.cfg.use_true_curr_bspn and not self.cfg.use_true_bspn_for_ctr_eval:
                            bspn = turn['bspn_gen']
                        else:
                            bspn = turn['bspn']
                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspan_to_constraint_dict(bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if  ven not in venue_offered[domain]:
                                # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            if 'booked' in turn['pointer'] or 'ok' in turn['pointer']:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')
                            # provided_requestables[domain].append('reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts

    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for s in true_goal[domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append("reference")

            for s, v in true_goal[domain]['info'].items():
                s_,v_ = clean_slot_values(domain, s,v, self.mapping_pair_path)
                if len(v_.split())>1:
                    v_ = ' '.join([token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]["informable"][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']
        return goal
