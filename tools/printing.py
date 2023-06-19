import os
import matplotlib.pyplot as plt


__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['print_baseline_results', 'print_mixup_ict_results',
           'print_ict_only_results']


def print_baseline_results(ep, loss_tr, loss_val, tr_acc, val_acc, time_):

    log_output = \
        '  -- Epoch: {ep: 04d} | ' \
        'Loss(tr/val): {l_tr: 7.4f} /{l_val: 7.4f} | ' \
        'Accuracy(tr/val): {ac_tr: 7.4f} /{ac_val: 7.4f} | ' \
        'Time: {t: 5.2f}'.format(
            ep=ep,
            l_tr=loss_tr,
            l_val=loss_val,
            ac_tr=tr_acc,
            ac_val=val_acc,
            t=time_
        )
    print(log_output, flush=True)


def print_mixup_ict_results(ep, loss_tr, loss_val, loss_domain, tr_acc, val_acc,
                            acc_domain, time_, loss_consistency=None):

    if loss_consistency:
        log_output = \
            '  -- Epoch: {ep: 04d} | ' \
            'Loss(tr/val): {l_tr: 6.3f} /{l_val: 6.3f} | ' \
            'Accuracy(tr/val): {ac_tr: 6.3f} /{ac_val: 6.3f} | ' \
            'Domain(loss/acc): {l_da: 6.3f} /{ac_da: 6.3f} | ' \
            'Consistency: {l_cons: 6.4f} | ' \
            'Time: {t: 5.2f}'.format(
                ep=ep,
                l_tr=loss_tr,
                l_val=loss_val,
                ac_tr=tr_acc,
                ac_val=val_acc,
                l_da=loss_domain,
                ac_da=acc_domain,
                l_cons=loss_consistency,
                t=time_
            )
    else:
        log_output = \
            '  -- Epoch: {ep: 04d} | ' \
            'Loss(tr/val): {l_tr: 6.3f} /{l_val: 6.3f} | ' \
            'Accuracy(tr/val): {ac_tr: 6.3f} /{ac_val: 6.3f} | ' \
            'Domain(loss/acc): {l_da: 6.3f} /{ac_da: 6.3f} | ' \
            'Time: {t: 5.2f}'.format(
                ep=ep,
                l_tr=loss_tr,
                l_val=loss_val,
                ac_tr=tr_acc,
                ac_val=val_acc,
                l_da=loss_domain,
                ac_da=acc_domain,
                t=time_
            )

    print(log_output, flush=True)


def print_ict_only_results(ep, loss_tr, loss_consistency, loss_val, tr_acc, val_acc, cons_weight,
                           time_):

    log_output = \
        '  -- Epoch: {ep:03d} | ' \
        'Loss(tr/val): {l_tr:5.3f} /{l_val: 5.3f}| ' \
        'Accuracy(tr/val): {ac_tr:5.3f} /{ac_val: 5.3f} | ' \
        'Consistency(l/w): {l_cons:5.4f} /{w_cons: 4.3f} | ' \
        'Time: {t:4.2f}'.format(
            ep=ep,
            l_tr=loss_tr,
            l_val=loss_val,
            ac_tr=tr_acc,
            ac_val=val_acc,
            l_cons=loss_consistency,
            w_cons=cons_weight,
            t=time_
        )

    print(log_output, flush=True)

# EOF
