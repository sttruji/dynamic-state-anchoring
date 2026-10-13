"""Parameter sharing, causal isolation and checkpoint gradients of memory injection."""
import copy
from dataclasses import replace
import tempfile
import unittest
import torch
from dsa.models.anchored_loop import AnchorConfig, AnchoredLoop
from dsa.models.backbone import BackboneAdapter, position_ids
from dsa.models.flt_memory import memory_block
from dsa.training.pilot import _state_digest

torch.set_num_threads(1)

class FLTMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.standard = AnchoredLoop(BackboneAdapter.tiny(259), AnchorConfig(loop_budget=3)).eval()
        self.flt = copy.deepcopy(self.standard)
        self.flt.config = replace(self.flt.config, recurrence_mode='flt_memory')
        self.question = torch.tensor([[10,11,12]])
        self.mask = torch.ones_like(self.question)

    def test_same_parameters_initialization_and_first_loop(self):
        self.assertEqual(_state_digest(self.standard), _state_digest(self.flt))
        self.assertEqual(sum(p.numel() for p in self.standard.parameters()), sum(p.numel() for p in self.flt.parameters()))
        a=self.standard(self.question,self.mask,loop_budget=1);b=self.flt(self.question,self.mask,loop_budget=1)
        torch.testing.assert_close(a.workspace,b.workspace,rtol=0,atol=0)
        torch.testing.assert_close(a.anchor,b.anchor,rtol=0,atol=0)

    def test_memory_block_matches_pretrained_block_when_queries_equal(self):
        block=self.standard.backbone.model.transformer.h[0]
        hidden=torch.randn(1,8,32)
        mask=torch.zeros(1,1,1,8)
        expected=block(hidden,attention_mask=mask,use_cache=False)[0]
        actual=memory_block(block,hidden,hidden[:,3:],mask,3)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)

    def test_previous_memory_reaches_each_block_without_changing_question_positions(self):
        for block in self.flt.backbone.model.transformer.h:
            hidden=torch.randn(1,8,32)
            previous=torch.randn(1,5,32)
            mask=torch.zeros(1,1,1,8)
            a=memory_block(block,hidden,previous,mask,3)
            b=memory_block(block,hidden,previous+torch.randn_like(previous),mask,3)
            torch.testing.assert_close(a[:,:3],b[:,:3],rtol=0,atol=0)
            self.assertFalse(torch.allclose(a[:,3:],b[:,3:]))

    def test_static_anchor_and_causal_padding(self):
        self.flt.config=replace(self.flt.config,write_mode='static')
        a=self.flt(self.question,self.mask)
        self.assertTrue(torch.equal(a.anchors[:,0],a.anchor))
        b=self.flt(torch.tensor([[0,0,10,11,12]]),torch.tensor([[0,0,1,1,1]]))
        torch.testing.assert_close(a.workspace,b.workspace,atol=3e-6,rtol=3e-5)
        c=self.flt(torch.tensor([[10,11,12,0,0],[10,21,31,41,51]]),torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]))
        torch.testing.assert_close(a.workspace[0],c.workspace[0],atol=3e-6,rtol=3e-5)

    def test_explicit_and_readout_paths_stay_ordinary_gpt2(self):
        ids=torch.tensor([[5,6,7,8]])
        embeds=self.standard.backbone.embeddings(ids)
        a=self.standard.backbone.forward_embeds(embeds,torch.ones_like(ids),position_ids(torch.ones_like(ids)))
        b=self.flt.backbone.forward_embeds(embeds,torch.ones_like(ids),position_ids(torch.ones_like(ids)))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        loop=self.standard(self.question,self.mask)
        torch.testing.assert_close(self.standard.answer_logits(loop,ids,torch.ones_like(ids)),
                                   self.flt.answer_logits(loop,ids,torch.ones_like(ids)),rtol=0,atol=0)

    def test_end_to_end_gradients_and_checkpoint_recomputation_agree(self):
        direct=copy.deepcopy(self.flt).train()
        # Nonzero dropout also checks RNG preservation on checkpoint recompute.
        for module in direct.modules():
            if isinstance(module,torch.nn.Dropout): module.p=.1
        checked=copy.deepcopy(direct)
        checked.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        def gradients(model):
            torch.manual_seed(12)
            loop=model(self.question,self.mask)
            logits=model.answer_logits(loop,torch.tensor([[15,16,17]]),torch.ones(1,3,dtype=torch.long))
            torch.nn.functional.cross_entropy(logits[:,-1],torch.tensor([22])).backward()
            return {name:p.grad for name,p in model.named_parameters()}
        a,b=gradients(direct),gradients(checked)
        for name in a:
            if a[name] is None:self.assertIsNone(b[name])
            else:
                self.assertTrue(torch.isfinite(b[name]).all(),name)
                torch.testing.assert_close(a[name],b[name],atol=2e-5,rtol=2e-4,msg=name)
        self.assertGreater(a['initial_workspace'].norm().item(),0)
        self.assertGreater(a['gate_head.weight'].norm().item(),0)

    def test_save_load_retains_variant_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            from dsa.data.tokenize import TinyTokenizer
            self.flt.save_pretrained(temp,TinyTokenizer())
            restored=AnchoredLoop.from_pretrained(temp).eval()
            self.assertEqual(restored.config.recurrence_mode,'flt_memory')
            a=self.flt(self.question,self.mask);b=restored(self.question,self.mask)
            torch.testing.assert_close(a.workspace,b.workspace,atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
