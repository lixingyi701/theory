#!/usr/bin/env python3
"""Shared one-attempt kernels; a task's latent success offset persists across trials."""
import numpy as np

OUTCOMES = ['S', 'E', 'C', 'B', 'U', 'A']
FAMILIES = ['constant', 'error_lifetime', 'information_order']
SCALES = [0., 1.5]
OFFSETS = np.asarray([-1., 0., 1.])
PRIOR = np.asarray([1/6, 2/3, 1/6])


def features(state, family):
    """Entries are oldest first; B explicitly contains both incompatible rules."""
    if family not in FAMILIES or len(state) > 3:
        raise ValueError('Unsupported family/window')
    if any(s['type'] not in OUTCOMES[1:]+['N'] or type(s['native']) is not bool for s in state):
        raise ValueError('Invalid entry type/origin')
    if family == 'constant':
        return [1.]
    aligned = [None]*(3-len(state))+state
    e = [float(s is not None and s['type'] in ['E', 'B']) for s in aligned]
    c = [float(s is not None and s['type'] in ['C', 'B']) for s in aligned]
    native = [float(s is not None and s['native']) for s in aligned]
    a = sum(s['type'] == 'A' for s in state)/3
    base = [1., len(state)/3, sum(native)/3, a]
    if family == 'error_lifetime':
        return base+[float(any(e)), max([i+1 for i,v in enumerate(e) if v] or [0])/3]
    return base+e+c+[float(any(e) and any(c))]


def softmax(z):
    z = z-np.max(z, axis=-1, keepdims=True)
    v = np.exp(z)
    return v/v.sum(axis=-1, keepdims=True)


def bfgs(function, initial):
    """Small deterministic inverse-BFGS; analytic gradient, Armijo line search."""
    value = np.asarray(initial, dtype=float).copy()
    h = np.eye(len(value)); loss, grad = function(value)
    for iteration in range(500):
        if np.max(np.abs(grad)) < 1e-5:
            return value, {'iterations': iteration, 'max_gradient': float(np.max(np.abs(grad)))}
        direction = -h@grad
        if direction@grad >= 0:
            h = np.eye(len(value)); direction = -grad
        step = 1.
        while step > 2**-35:
            candidate = value+step*direction
            next_loss, next_grad = function(candidate)
            if next_loss <= loss+1e-4*step*(grad@direction):
                break
            step *= .5
        if step <= 2**-35:
            raise RuntimeError('Model line search failed')
        delta = candidate-value; y = next_grad-grad; dot = y@delta
        if dot > 1e-10:
            rho = 1/dot; identity = np.eye(len(value))
            h = (identity-rho*np.outer(delta,y))@h@(identity-rho*np.outer(y,delta))+rho*np.outer(delta,delta)
        else:
            h = np.eye(len(value))
        value, loss, grad = candidate, next_loss, next_grad
    raise RuntimeError('Model optimizer did not converge')


def success_objective(beta, x, y, groups, scale):
    logits = x@beta[:,None]+scale*OFFSETS[None,:]
    p = 1/(1+np.exp(-np.clip(logits,-700,700)))
    logs = y[:,None]*(-np.logaddexp(0,-logits))+(1-y[:,None])*(-np.logaddexp(0,logits))
    loss = 0.; gradient = np.zeros_like(beta)
    for indices in groups:
        weights = softmax(np.log(PRIOR)+logs[indices].sum(axis=0))
        joint = np.log(PRIOR)+logs[indices].sum(axis=0)
        m = joint.max(); loss -= m+np.log(np.exp(joint-m).sum())
        gradient += x[indices].T@((p[indices]-y[indices,None])@weights)
    # Fixed ridge; symmetric half-count at zero-feature intercept keeps all-success folds finite.
    penalty = beta.copy(); penalty[0] = 0.
    loss += .5*(penalty@penalty)+.5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0])
    gradient += penalty; gradient[0] += 1/(1+np.exp(-beta[0]))-.5
    return float(loss), gradient


def fit(rows, family, scale):
    x = np.asarray([features(r['state'],family) for r in rows]); y = np.asarray([r['outcome']=='S' for r in rows],float)
    tasks = sorted({r['task_id'] for r in rows})
    groups = [np.asarray([i for i,r in enumerate(rows) if r['task_id']==t]) for t in tasks]
    beta, audit_p = bfgs(lambda b: success_objective(b,x,y,groups,scale),np.zeros(x.shape[1]))
    failures = [i for i,r in enumerate(rows) if r['outcome']!='S']
    xx = np.r_[x[failures],np.r_[1.,np.zeros(x.shape[1]-1)][None,:].repeat(5,axis=0)]
    yy = np.r_[np.eye(5)[[OUTCOMES[1:].index(rows[i]['outcome']) for i in failures]],np.eye(5)]
    weights = np.r_[np.ones(len(failures)),np.full(5,.5)]
    shape = (x.shape[1],4)
    def objective(flat):
        b = flat.reshape(shape); probs = softmax(np.c_[xx@b,np.zeros(len(xx))])
        penalty = b.copy(); penalty[0] = 0
        loss = -np.sum(weights[:,None]*yy*np.log(np.clip(probs,1e-300,1)))+.5*np.sum(penalty**2)
        gradient = xx.T@(weights[:,None]*(probs[:,:4]-yy[:,:4]))+penalty
        return float(loss),gradient.ravel()
    coef, audit_g = bfgs(objective,np.zeros(np.prod(shape)))
    return {'family':family,'scale':scale,'success_coefficients':beta.tolist(),
            'write_coefficients_reference_A':coef.reshape(shape).tolist(),'optimizer':{'p':audit_p,'G':audit_g},
            'latent_prior':PRIOR.tolist(),'latent_offsets':OFFSETS.tolist(),
            'task_effect_scope':'Persistent success intercept; conditional write kernel shared across latent task classes'}


def component_kernel(model, state):
    x = np.asarray(features(state,model.get('success_family',model['family'])))
    xx = np.asarray(features(state,model.get('write_family',model['family'])))
    p = 1/(1+np.exp(-np.clip(x@np.asarray(model['success_coefficients'])+model['scale']*OFFSETS,-700,700)))
    g = softmax(np.r_[xx@np.asarray(model['write_coefficients_reference_A']),0.])
    return np.c_[p,(1-p[:,None])*g[None,:]]


def predict_sequence(model, rows):
    """Only preceding observed attempts of this trajectory update its task prior."""
    weights = PRIOR.copy(); predictions = []
    for row in rows:
        kernel = component_kernel(model,row['state']); probability = weights@kernel
        predictions.append({k:float(v) for k,v in zip(OUTCOMES,probability)})
        weights *= kernel[:,OUTCOMES.index(row['outcome'])]; weights /= weights.sum()
    return predictions


def budget_probability(model, state, budget):
    """Integrate over one persistent latent draw, never redraw it after failure."""
    def recurse(current,left,component):
        if left == 0:
            return 0.
        kernel = component_kernel(model,current)[component]
        return kernel[0]+sum(kernel[i+1]*recurse((current+[{'type':kind,'native':True}])[-3:],left-1,component)
                            for i,kind in enumerate(OUTCOMES[1:]))
    return float(sum(w*recurse(state,budget,k) for k,w in enumerate(PRIOR)))


def scores(prediction, outcome):
    p = np.asarray([prediction[k] for k in OUTCOMES]); y = np.eye(6)[OUTCOMES.index(outcome)]
    return {'joint_brier':float(np.sum((p-y)**2)),'success_brier':float((p[0]-y[0])**2),
            'log_loss':float(-np.log(max(p[OUTCOMES.index(outcome)],1e-300)))}


def compare(rows):
    scenes = sorted({r['scene_id'] for r in rows}); folds = []
    for scene in scenes:
        train = [r for r in rows if r['scene_id']!=scene]
        test = [r for r in rows if r['scene_id']==scene]
        for family in FAMILIES:
            for scale in SCALES:
                model = fit(train,family,scale); trajectory_scores = []; write_scores = []
                for trajectory in sorted({r['trajectory_id'] for r in test}):
                    sequence = sorted([r for r in test if r['trajectory_id']==trajectory],key=lambda r:r['trial'])
                    values = [scores(p,r['outcome']) for p,r in zip(predict_sequence(model,sequence),sequence)]
                    trajectory_scores.append({key:float(np.mean([v[key] for v in values])) for key in values[0]})
                    failures = [r for r in sequence if r['outcome']!='S']
                    if failures:
                        losses=[]
                        for row in failures:
                            kernel=component_kernel(model,row['state'])[0]
                            g=kernel[1:]/(1-kernel[0]); target=np.eye(5)[OUTCOMES[1:].index(row['outcome'])]
                            losses.append(float(np.sum((g-target)**2)))
                        write_scores.append(float(np.mean(losses)))
                folds.append({'scene_id':scene,'family':family,'scale':scale,
                    **{key:float(np.mean([v[key] for v in trajectory_scores])) for key in trajectory_scores[0]},
                    'write_brier':float(np.mean(write_scores)) if write_scores else None})
    summaries = []
    for family in FAMILIES:
        for scale in SCALES:
            selected = [r for r in folds if r['family']==family and r['scale']==scale]
            summaries.append({'family':family,'scale':scale,**{k:float(np.mean([r[k] for r in selected]))
                              for k in ['joint_brier','success_brier','log_loss']},
                              'write_brier':float(np.mean([r['write_brier'] for r in selected if r['write_brier'] is not None]))})
    def select(metric,choices):
        best=min(choices,key=lambda r:r[metric]); eligible=[]
        reference={r['scene_id']:r[metric] for r in folds if r['family']==best['family'] and r['scale']==best['scale'] and r[metric] is not None}
        for summary in choices:
            delta=np.asarray([r[metric]-reference[r['scene_id']] for r in folds if r['family']==summary['family'] and
                              r['scale']==summary['scale'] and r['scene_id'] in reference and r[metric] is not None])
            se=float(np.std(delta,ddof=1)/np.sqrt(len(delta)))
            summary[metric+'_paired_excess_se']=se
            if float(delta.mean())<=se+1e-12: eligible.append(summary)
        return min(eligible,key=lambda r:(FAMILIES.index(r['family']),SCALES.index(r['scale'])))
    selected_p=select('success_brier',summaries)
    selected_g=select('write_brier',[r for r in summaries if r['scale']==0.])
    fitted = [fit(rows,f,s) for f in FAMILIES for s in SCALES]
    p_model=next(m for m in fitted if m['family']==selected_p['family'] and m['scale']==selected_p['scale'])
    g_model=next(m for m in fitted if m['family']==selected_g['family'] and m['scale']==0.)
    selected={**p_model,'success_family':p_model['family'],'write_family':g_model['family'],
              'write_coefficients_reference_A':g_model['write_coefficients_reference_A'],
              'optimizer':{'p':p_model['optimizer']['p'],'G':g_model['optimizer']['G']}}
    return {'folds':folds,'scores':summaries,'selected_success':selected_p,'selected_write':selected_g,
            'selected_model':selected,'all_fitted_models':fitted,
            'scope':'Development calibration with held-scene model selection; prospective validation not yet collected'}
