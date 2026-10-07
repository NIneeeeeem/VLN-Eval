"""Explicitly captured upstream Habitat measurements.

These are diagnostic values emitted by the selected upstream task, not
reimplementations of or a claim of parity with its challenge leaderboard.
"""
from nav_eval.contracts import ContractError, finite_number
from nav_eval.evaluation.metrics import MissingEvidence


class CapturedMetric:
    required_fields = ("upstream_metrics_by_tick",)
    path = ()

    def compute(self, _record, evidence, _parameters):
        samples = evidence.get("upstream_metrics_by_tick")
        if not isinstance(samples, list) or not samples:
            raise MissingEvidence("upstream captured metric samples are unavailable")
        value = samples[-1]
        for field in self.path:
            if not isinstance(value, dict) or field not in value:
                raise MissingEvidence(f"upstream measurement {'.'.join(self.path)} is unavailable")
            value = value[field]
        if isinstance(value, bool):
            return float(value)
        try:
            return finite_number(value, self.name)
        except ContractError as error:
            raise MissingEvidence(f"upstream measurement {'.'.join(self.path)} is not finite scalar evidence") from error


class CapturedUpstreamSuccess(CapturedMetric):
    name, path = "upstream_habitat.captured_success", ("success",)

class CapturedUpstreamSPL(CapturedMetric):
    name, path = "upstream_habitat.captured_spl", ("spl",)


class GoatCompositeSuccess(CapturedMetric):
    name, path = "goat.captured_composite_success", ("success", "composite_success")


class GoatPartialSuccess(CapturedMetric):
    name, path = "goat.captured_partial_success", ("success", "partial_success")


class GoatObjectSuccess(CapturedMetric):
    name, path = "goat.captured_object_success", ("success", "object_success")


class GoatImageSuccess(CapturedMetric):
    name, path = "goat.captured_image_success", ("success", "image_success")


class GoatDescriptionSuccess(CapturedMetric):
    name, path = "goat.captured_description_success", ("success", "description_success")


class GoatCompositeSPL(CapturedMetric):
    name, path = "goat.captured_composite_spl", ("spl", "composite_spl")


class MultiONProgress(CapturedMetric):
    name, path = "multion.captured_progress", ("progress",)

    def compute(self, record, evidence, parameters):
        if evidence.get("provenance", {}).get("simulator_version") == "0.1.4":
            return MultiONLegacyProgress().compute(record, evidence, parameters)
        return super().compute(record, evidence, parameters)


class MultiONPPL(CapturedMetric):
    name, path = "multion.captured_ppl", ("ppl",)

    def compute(self, record, evidence, parameters):
        if evidence.get("provenance", {}).get("simulator_version") == "0.1.4":
            return MultiONLegacyPPL().compute(record, evidence, parameters)
        return super().compute(record, evidence, parameters)


class MultiONLegacyProgress(CapturedMetric):
    name, path = "multion.captured_progress", ("percentage_success",)


class MultiONLegacyPPL(CapturedMetric):
    name, path = "multion.captured_ppl", ("pspl",)


class MultiONSuccess(CapturedMetric):
    name, path = "multion.captured_success", ("success",)


class MultiONMSPL(CapturedMetric):
    name, path = "multion.captured_mspl", ("mspl",)
