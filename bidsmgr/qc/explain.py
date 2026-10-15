"""What each quality measure and QC plot means, and how to read it.

One entry per measure of the anatomical and diffusion checks
(``bidsmgr.qc.anat``, ``bidsmgr.qc.dwi``), per group of the measures table,
per QC plot under a time course (``viz.compute.qc.QC_ROWS`` and
``DWI_QC_ROWS``) and per BOLD quality map. ``short`` is the hover text; the
other fields fill the info popup. Every statement about how a value is made
follows the code that makes it; thresholds name their default and the
Settings section that changes them.

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import Optional


@dataclass(frozen=True)
class Explanation:
    #: The name on screen.
    title: str
    #: One or two sentences for the hover text: what it is, and which way is better.
    short: str
    #: What it measures, in plain words.
    measures: str
    #: How BIDS Manager computes it: which image, which mask, the formula in words.
    computed: str
    #: How to read it: the direction, what is typical, what to compare it with.
    reading: str
    #: What pushes it the wrong way: artefacts, acquisition, physiology, processing.
    causes: str
    #: What it does NOT tell, and the pitfalls.
    caveats: str
    #: Its relation to MRIQC: the same definition, how it differs, or "".
    mriqc: str = ""
    #: Papers it follows, short form ("Power et al. 2012, NeuroImage").
    references: tuple[str, ...] = ()


# References used more than once.
_MRIQC = "Esteban et al. 2017, PLoS One"
_DIETRICH = "Dietrich et al. 2007, J Magn Reson Imaging"
_HENKELMAN = "Henkelman 1985, Med Phys"
_GANZETTI = "Ganzetti et al. 2016, Front Neuroinform"
_ATKINSON = "Atkinson et al. 1997, IEEE Trans Med Imaging"
_MORTAMET = "Mortamet et al. 2009, Magn Reson Med"
_SHEHZAD = "Shehzad et al. 2015, Front Neurosci (Neuroinformatics 2015 conference abstract)"
_FORMAN = "Forman et al. 1995, Magn Reson Med"
_SLED = "Sled et al. 1998, IEEE Trans Med Imaging"
_CRUM = "Crum et al. 2006, IEEE Trans Med Imaging"
_POWER12 = "Power et al. 2012, NeuroImage"
_POWER17 = "Power 2017, NeuroImage"
_AFYOUNI = "Afyouni and Nichols 2018, NeuroImage"
_COX = "Cox 1996, Comput Biomed Res"
_KRUEGER = "Krueger and Glover 2001, Magn Reson Med"
_TRIANTAFYLLOU = "Triantafyllou et al. 2005, NeuroImage"
_VERAART_NI = "Veraart et al. 2016, NeuroImage"
_VERAART_MRM = "Veraart, Fieremans and Novikov 2016, Magn Reson Med"
_YEH = "Yeh et al. 2019, NeuroImage"
_GUDBJARTSSON = "Gudbjartsson and Patz 1995, Magn Reson Med"
_BASSER94 = "Basser, Mattiello and LeBihan 1994, Biophys J"
_BASSER96 = "Basser and Pierpaoli 1996, J Magn Reson B"
_LEEMANS = "Leemans and Jones 2009, Magn Reson Med"
_ANDERSSON_EDDY = "Andersson and Sotiropoulos 2016, NeuroImage"
_ANDERSSON_OLR = "Andersson et al. 2016, NeuroImage"
_VOS = "Vos et al. 2017, Magn Reson Med"
_JONES99 = "Jones, Horsfield and Simmons 1999, Magn Reson Med"
_JEURISSEN = "Jeurissen, Leemans and Sijbers 2014, Med Image Anal"


# ---------------------------------------------------------------------------
# Anatomical images: the groups of the measures table
# ---------------------------------------------------------------------------

_ANAT_GROUPS: dict[str, Explanation] = {
    "group.anat.noise": Explanation(
        title="Noise and contrast",
        short=("How large the signal is against the noise, and how well grey and white "
               "matter are told apart. Compare only with images of the same protocol."),
        measures=(
            "Three families. The within-tissue SNRs (CSF, grey matter, white matter and "
            "their mean) compare each tissue's typical intensity with the spread inside "
            "that same tissue. The SNRs against the air compare tissue intensity with the "
            "noise in the background around the head. CNR and CJV measure how far apart "
            "grey and white matter are against their spreads."),
        computed=(
            "The brain comes from mindgrab or the registered template, the tissue classes "
            "from the robust_tissue network or an EM fit with the template's tissue maps "
            "as priors, as Settings > Quality control > Methods chooses. The classes are "
            "then estimated again at the image's own resolution on bias-corrected "
            "intensities, so each tissue's median and spread come from soft memberships "
            "(probabilities), not from a hard mask. The air is the region outside the head "
            "by a small margin, above the plane through the glabella and the inion, "
            "without zero fill."),
        reading=(
            "Within-tissue SNR is signal over everything that varies inside a tissue: "
            "thermal noise, but also real biological variation, residual bias field, "
            "partial volume at the class borders and misclassified voxels. It is therefore "
            "an upper bound on the noise, not a pure noise measure, and the three tissues "
            "differ for reasons that have nothing to do with quality. On a T1w, white "
            "matter is bright, large and homogeneous, so its spread comes closest to the "
            "noise and its SNR is the most informative of the three; grey matter is a thin "
            "ribbon full of border voxels whose intensity also differs between cortex and "
            "deep nuclei, so its SNR is lower; CSF is the darkest class and mostly thin "
            "sulci, so its SNR is low even in an excellent scan. On a T2w the order of "
            "brightness reverses (CSF bright, white matter dark) and the expectations "
            "change with it. The SNRs against the air use the background noise, which is "
            "closer to the thermal noise, but they are valid only where the air is real "
            "and untouched. CNR and CJV say whether the grey-white boundary survives the "
            "noise; they depend strongly on the sequence, the field strength and the "
            "participant's age. The right comparison is with scans of the same contrast, "
            "sequence, field strength, coil and voxel size, which is what the 'In this "
            "dataset' column gives."),
        causes=(
            "Small voxels, short scans and high acceleration raise the noise; motion blurs "
            "the tissues together and puts ghosts in the air; residual bias field and "
            "partial volume widen every tissue's spread."),
        caveats=(
            "Absolute values do not travel between protocols: image SNR grows with voxel "
            "volume, parallel imaging (GRAPPA, SENSE) lowers it and makes the noise vary "
            "across the image, and reconstruction filters change it. The 'In this dataset' "
            "column is a robust z, (value - median) / (1.4826 x MAD), among the dataset's "
            "saved results with the same suffix and acq label (three or more): 'typical' "
            "within 2, coloured as a warning from 3 in the worse direction. Results made "
            "with the networks and with the fast fallback differ, so compare results made "
            "with the same methods. The air measures are not computed for defaced, skull-"
            "stripped or zero-filled images, and no tissue measure is computed for PDw, "
            "T2starw or thin slabs."),
        mriqc=(
            "The within-tissue SNRs, CNR and CJV share MRIQC's formulas (snr_*, cnr, cjv) "
            "and the SNRs against the air follow the same method as snrd_*. Values still "
            "differ from MRIQC's through the segmentation and the bias correction, so do "
            "not mix the two in one comparison."),
        references=(_DIETRICH, _GANZETTI, _MRIQC),
    ),
    "group.anat.artefacts": Explanation(
        title="Artefacts",
        short=("Signal where there should be only noise: in the air around the head. "
               "Lower is better for all but FBER."),
        measures=(
            "Ghosts, ringing, motion and wrap-around put signal outside the head, where a "
            "clean scan has only noise. EFC, FBER, QI1, QI2 and the ghost ratio each look "
            "at that background in a different way."),
        computed=(
            "The air is found on the working grid (about 2 mm): outside the head mask by "
            "about two working voxels, above the plane through the glabella and the inion "
            "(below it the face, neck and shoulders move and fold normally), and without "
            "zero fill. EFC uses every voxel except the zero fill and FBER the whole "
            "background, at the image's resolution; QI1 and the ghost ratio use the "
            "measured air on the working grid, QI2 the measured air at the image's "
            "resolution."),
        reading=(
            "QI1 says how much of the air holds structured signal, and the 'Artefacts in "
            "the air' layer shows where. QI2 says whether what is left looks like noise. "
            "The ghost ratio checks the one place where an echo-planar ghost falls. EFC "
            "and FBER summarise how the image's energy is split between head and "
            "background. A motion-corrupted scan usually moves several together: QI1, QI2 "
            "and EFC up, FBER down; one measure out of line with the others is more often "
            "a property of the field of view or of the background handling than an "
            "artefact. The findings add what a number cannot: structured signal in more "
            "than 0.5 % of the air by default (Settings > Quality control > Anatomical "
            "images), and wrap-around when the head reaches the edge of the field of view "
            "along the phase-encoding axis and more than 2 % of the air in the outer "
            "tenth at the two ends of that axis holds structured signal."),
        causes=(
            "Head motion, pulsation and flow, Nyquist ghosting, wrap-around from a small "
            "field of view, RF interference, and a scanner that filters or rescales the "
            "background."),
        caveats=(
            "None is computed when the air is not real: a defaced image (the sidecar's "
            "record, or a blank face region), a skull-stripped one, one with too little "
            "air, or one whose air is mostly exactly zero. Those images are reported as "
            "not measured, with the reason, rather than given a number that would "
            "describe the defacing instead of the scan. Multi-channel coils and parallel "
            "imaging change the background noise itself, which moves QI2 and EFC without "
            "any artefact."),
        mriqc=(
            "EFC, FBER and QI2 share MRIQC's formulas; QI1 follows the definition MRIQC "
            "intended; the ghost ratio has no anatomical equivalent in MRIQC."),
        references=(_MORTAMET, _ATKINSON),
    ),
    "group.anat.tissues": Explanation(
        title="Tissues",
        short=("What the segmentation found: how the intensity range is used, how uniform "
               "the image is, the tissue shares, partial volume, and agreement with the "
               "template."),
        measures=(
            "These describe the tissue classes and the bias field rather than the noise. "
            "They flag a failed segmentation or registration, a strongly non-uniform image, "
            "and blur."),
        computed=(
            "The brain comes from mindgrab or the registered template, the classes from "
            "the robust_tissue network (grey and white matter; CSF is the rest of the "
            "brain) or an EM fit with the template's tissue maps as priors, as Settings > "
            "Quality control > Methods chooses; a method whose tool cannot run falls back "
            "to the fast one and a finding says so. The bias field is fitted to what the "
            "classes do not explain. The network segments T1w, T2w and FLAIR; the fallback "
            "only T1w and T2w."),
        reading=(
            "The tissue fractions and the template overlaps are mainly checks on the "
            "masks: an outlier among the dataset's images usually means a segmentation or "
            "a registration that went wrong, and the 'Tissue classes' and 'Brain mask' "
            "layers show it. Intensity non-uniformity says how much the bias correction "
            "had to do, and white matter to maximum how the intensity range is used. "
            "Partial volume rises with blur, thick voxels and noise. Read all of them "
            "against images of the same contrast in the dataset; absolute values depend "
            "on the methods used."),
        causes=(
            "A failed brain mask or registration, strong coil sensitivity, thick voxels, "
            "motion blur, and anatomy far from the adult template."),
        caveats=(
            "Not computed on PDw or T2starw images, on FLAIR without the tissue network, "
            "or on slabs thinner than 70 mm by default (Settings > Quality control > "
            "Anatomical images), which are not registered. The fast fallback's masks are "
            "approximate, and so are its tissue measures."),
        references=(_SLED,),
    ),
    "group.anat.coverage": Explanation(
        title="Coverage and smoothness",
        short=("Whether the field of view holds the whole brain, and how smooth the image "
               "is along each axis."),
        measures=(
            "The share of the brain that the field of view cuts off, from the registered "
            "template, and the smoothness along each voxel axis and its mean, from the "
            "image's own correlation between neighbours."),
        computed=(
            "The template's brain is carried through the registration into the image's "
            "voxel grid and the part landing outside is counted. Smoothness is Forman's "
            "estimate from neighbouring voxels inside the brain, at the image's own "
            "resolution, in millimetres."),
        reading=(
            "A cut brain is a silent reason to exclude a scan: a missing cerebellum or "
            "vertex cannot be recovered, and the finding names the side. Smoothness is "
            "read against the protocol's other scans: a rise means blur, resampling or "
            "larger voxels, and a fall can mean more noise rather than a sharper image. A "
            "slab (a field of view thinner than 120 mm by default) covers part of the "
            "brain on purpose and is reported as coverage, not as a cut; below 70 mm by "
            "default the template is not registered and the cut is not computed (both "
            "Settings > Quality control > Anatomical images)."),
        causes=(
            "A field of view placed too low or too high, a small field of view, motion "
            "blur, reconstruction filtering, interpolation and resampling."),
        caveats=(
            "The cut is only as good as the registration, and an uncertain registration "
            "is reported as a finding. Smoothness includes the brain's own structure and "
            "is in millimetres, so it does not compare across voxel sizes."),
        references=(_FORMAN,),
    ),
    "group.anat.header": Explanation(
        title="Header",
        short=("What the stored file says about the data: voxels clipped at the top of "
               "the range, and (as findings) the orientation and voxel shape."),
        measures=(
            "Saturated voxels, plus findings from the NIfTI header: a qform and an sform "
            "that disagree, an oblique acquisition, and voxels much longer along one axis "
            "than another."),
        computed=(
            "Saturation is counted inside the head at the image's resolution. The qform "
            "and sform are compared element by element (a difference above 0.001 is "
            "reported); the tilt of the voxel axes from the scanner's axes is measured (a "
            "note above 5 degrees); the voxel sizes are compared (a note when one is more "
            "than twice another)."),
        reading=(
            "Saturation above 0.5 % by default is reported (Settings > Quality control > "
            "Anatomical images). A qform and an sform that disagree matter: tools that "
            "read one or the other place the image differently, which can misalign it "
            "with other images. An oblique acquisition and thick voxels are notes, not "
            "faults: the first matters to tools that ignore the orientation, the second "
            "makes the tissue measures less reliable."),
        causes=(
            "Receiver or reconstruction scaling that overflowed the stored range, a "
            "conversion or a tool that rewrote the header, and the acquisition's own "
            "planning."),
        caveats=(
            "The header findings describe the file, not the image's quality: a correct "
            "but oblique image is fine for tools that use the orientation."),
    ),
}


# ---------------------------------------------------------------------------
# Anatomical images: the measures
# ---------------------------------------------------------------------------

_TISSUE_SNR_COMPUTED = (
    "Bias-corrected intensities of the brain's voxels at the image's own resolution (up "
    "to a million sampled), each weighted by its probability of belonging to the tissue, "
    "from the class estimate made again at that resolution. The value is the weighted "
    "median over the weighted standard deviation, the SD multiplied by sqrt(n/(n-1)) with "
    "n the sum of the weights (a negligible correction at these sizes).")

_ANAT_NOISE: dict[str, Explanation] = {
    "anat.snr_wm": Explanation(
        title="SNR in white matter",
        short=("White matter's typical intensity over the spread of intensities inside "
               "white matter. Higher is better; on a T1w it is the most informative of "
               "the three tissue SNRs."),
        measures=(
            "How large white matter's signal is compared with how much its intensity "
            "varies from voxel to voxel inside white matter. The spread holds thermal "
            "noise plus residual bias field, partial volume at the border with grey "
            "matter, real variation within the tissue and misclassified voxels: it is "
            "signal over everything that varies inside the tissue."),
        computed=_TISSUE_SNR_COMPUTED,
        reading=(
            "Higher is better. On a T1w, white matter is bright, large and homogeneous, "
            "so its spread comes closest to the noise: of the three tissue SNRs this one "
            "tracks image quality best. On a T2w white matter is the darkest tissue, so "
            "its value is lower there and is read against other T2w images. Compare only "
            "with scans of the same contrast, sequence, field strength, coil and voxel "
            "size, which is what the 'In this dataset' column does."),
        causes=(
            "Thermal noise (small voxels, short scans, high acceleration), motion that "
            "blurs grey matter into white matter and adds ghosts, bias field the "
            "correction left behind, and white-matter lesions that widen the class. A "
            "segmentation that puts grey-matter or CSF voxels in the class lowers it too."),
        caveats=(
            "An upper bound on the noise, not a pure noise measure: a perfect scan of a "
            "brain with variable white matter still reads lower. The memberships follow "
            "intensity at the borders, so voxels far from the class's typical value are "
            "partly given to another class, which trims the spread; the value depends on "
            "how the classes are split, and results from the networks and from the fast "
            "fallback should not be mixed. Smoothing at reconstruction or any resampling "
            "lowers the spread and raises the value at the cost of sharpness."),
        mriqc=(
            "Same definition as MRIQC's snr_wm (median over SD with the n/(n-1) "
            "correction). Values differ from MRIQC's through the segmentation and the "
            "bias correction."),
        references=(_MRIQC,),
    ),
    "anat.snr_gm": Explanation(
        title="SNR in grey matter",
        short=("Grey matter's typical intensity over the spread inside grey matter. Higher "
               "is better, but it reads lower than white matter's for anatomical reasons, "
               "not because of noise."),
        measures=(
            "How large grey matter's signal is compared with the variation of intensity "
            "inside the grey-matter class: thermal noise plus partial volume, real "
            "variation within the tissue, residual bias and misclassified voxels."),
        computed=_TISSUE_SNR_COMPUTED,
        reading=(
            "Higher is better. Expect it below white matter's on a T1w: grey matter is a "
            "ribbon a few millimetres thick where most voxels border white matter or CSF, "
            "and its intensity differs between the cortex and deep nuclei such as the "
            "thalamus and basal ganglia, so partial volume and anatomy inflate its spread. "
            "Read it beside white matter's: a fall in both points at noise or motion, a "
            "fall in grey matter's alone at blur, thick voxels or the segmentation."),
        causes=(
            "Noise and motion, as for white matter, and more than for white matter blur "
            "and thick voxels, which raise partial volume along the cortical ribbon. Deep "
            "grey nuclei, whose intensity differs from the cortex's, widen the class."),
        caveats=(
            "Not a noise measure: in a sharp, quiet scan it is still limited by partial "
            "volume and anatomy. Voxels between grey and white matter are shared between "
            "the two classes by their probabilities, so the value depends on the "
            "segmentation and differs between the networks and the fast fallback."),
        mriqc=("Same definition as MRIQC's snr_gm; values differ through the segmentation "
               "and the bias correction."),
        references=(_MRIQC,),
    ),
    "anat.snr_csf": Explanation(
        title="SNR in CSF",
        short=("CSF's typical intensity over the spread inside CSF. On a T1w it is low "
               "even in an excellent scan and says little about quality."),
        measures=(
            "How large the CSF's signal is compared with the variation of intensity "
            "inside the CSF class."),
        computed=(
            _TISSUE_SNR_COMPUTED + " With the tissue network, CSF is everything inside the "
            "brain mask that the network does not call grey or white matter, held loosely "
            "so that intensity can move a border voxel to another class."),
        reading=(
            "Higher is better only against the same protocol. On a T1w CSF is the darkest "
            "class, so its median is small, and its mask is mostly thin sulci where nearly "
            "every voxel is mixed with grey matter, so its spread is large next to that "
            "median: a low value there is normal. On a T2w CSF is the brightest class and "
            "the value is higher, but flow and pulsation add variation of their own."),
        causes=(
            "Partial volume with grey matter (thick voxels, blur, motion); the extent of "
            "the brain mask, since extra-cerebral CSF, vessels or dura inside it widen the "
            "class; atrophy, which enlarges sulci and ventricles and changes the mix; and "
            "on a T1w the noise of magnitude images near zero intensity, which is not "
            "Gaussian."),
        caveats=(
            "On a T1w it says more about anatomy, partial volume and the segmentation "
            "than about the scanner. Because it is low, it pulls the mean of the tissues "
            "down. Do not expect it to agree with the grey- or white-matter SNR."),
        mriqc=("Same definition as MRIQC's snr_csf; values differ through the segmentation "
               "and the bias correction."),
        references=(_MRIQC,),
    ),
    "anat.snr_total": Explanation(
        title="SNR (mean of the tissues)",
        short=("The mean of the CSF, grey-matter and white-matter SNRs. Higher is better, "
               "but CSF's low value pulls it down."),
        measures="A one-number summary of the three within-tissue SNRs.",
        computed=("The plain mean of the three tissue SNRs; one that could not be computed "
                  "is left out."),
        reading=(
            "Higher is better, against scans of the same protocol. It averages three "
            "measures that differ for anatomical reasons, so when it moves, look at the "
            "three to see which one moved. On a T1w, CSF's low SNR keeps it well below "
            "white matter's."),
        causes=("Everything that lowers any of the three tissue SNRs: noise, motion, "
                "residual bias, partial volume and the segmentation."),
        caveats=(
            "Two of its three parts (CSF and grey matter) are dominated by partial volume "
            "and anatomy, so it is a weaker quality measure than white matter's SNR alone. "
            "It weighs the three tissues equally, whatever their volume."),
        mriqc="Same definition as MRIQC's snr_total (the mean of the three).",
        references=(_MRIQC,),
    ),
    "anat.snrd_wm": Explanation(
        title="SNR against the air (white matter)",
        short=("White matter's typical intensity over the noise measured in the air around "
               "the head (Dietrich's method). Higher is better; needs real, untouched air."),
        measures=(
            "How large the white-matter signal is compared with the scanner's noise as the "
            "background shows it. Its denominator holds no anatomy, partial volume or bias "
            "field, so it is closer to the thermal noise than the within-tissue SNR."),
        computed=(
            "0.6551364 times white matter's (bias-corrected, probability-weighted) median, "
            "over the standard deviation of the intensities in the air: outside the head by "
            "a margin, above the plane through the glabella and the inion, without zero "
            "fill, at the image's own resolution. The factor is for magnitude images: where "
            "there is no signal the noise follows a Rayleigh distribution, whose SD is "
            "about 0.655 times the SD of the underlying Gaussian noise."),
        reading=(
            "Higher is better, against the same protocol. When its assumptions hold it is "
            "higher than the within-tissue SNR, because the air's spread holds no tissue "
            "variation. When the two disagree strongly, artefacts in the air (which raise "
            "its spread) or a filtered background (which lowers it) are the usual reasons."),
        causes=(
            "Noise (small voxels, short scans, acceleration) lowers it, and so does "
            "anything in the air that is not noise: ghosts, wrap-around, flow artefacts "
            "and ringing raise the air's SD."),
        caveats=(
            "Valid only with real air: not computed on a defaced image (the sidecar's "
            "record, or more than 40 % of the face region exactly zero by default), on a "
            "skull-stripped image, with too little air in the field of view, or when more "
            "than 50 % of the air is exactly zero by default (both Settings > Quality "
            "control > Anatomical images). The 0.655 factor holds for single-channel "
            "magnitude noise. With multi-channel coils, parallel imaging (the noise then "
            "varies across the image, and the background no longer shows the brain's "
            "noise) or a vendor filter on the background, the value is unreliable."),
        mriqc=(
            "Same method as MRIQC's snrd_wm. BIDS Manager leaves it empty where the air is "
            "not real, where MRIQC still reports a number."),
        references=(_DIETRICH, _HENKELMAN),
    ),
    "anat.snrd_total": Explanation(
        title="SNR against the air (mean)",
        short=("Dietrich's SNR averaged over CSF, grey and white matter. Higher is better; "
               "needs real air."),
        measures="The tissues' typical intensities, averaged, over the noise measured in the air.",
        computed=(
            "The mean of the three Dietrich SNRs, 0.6551364 times each tissue's median "
            "over the air's SD. The denominator is the same for all three, so it equals "
            "0.655 times the mean of the three medians over the air's SD."),
        reading=(
            "Higher is better, against the same protocol. On a T1w the CSF median is small, "
            "so this reads below the white-matter value; on a T2w CSF is bright and raises "
            "it. Read it beside the white-matter value."),
        causes=("As for the white-matter value: noise, and anything in the air that is not "
                "noise."),
        caveats=(
            "The same conditions as the white-matter value: real, untouched air, and "
            "single-channel magnitude noise for the 0.655 factor to hold. Its tissue "
            "medians depend on the segmentation, so an unusual CSF class moves it."),
        mriqc="Same method as MRIQC's snrd_total (the mean over the three tissues).",
        references=(_DIETRICH,),
    ),
    "anat.cnr": Explanation(
        title="Contrast-to-noise ratio",
        short=("How far apart grey and white matter are, against the noise of the air and "
               "the spread of both tissues. Higher is better; needs real air."),
        measures=(
            "The contrast between grey and white matter in units of the combined noise "
            "and tissue variation: whether the boundary between them stands out."),
        computed=(
            "|median WM - median GM| / sqrt(SD_air^2 + SD_GM^2 + SD_WM^2). The tissue "
            "medians and SDs come from the bias-corrected, probability-weighted "
            "intensities; SD_air is the plain SD of the air's intensities, with no Rayleigh "
            "correction."),
        reading=(
            "Higher is better. The tissue SDs dominate the denominator and already hold "
            "noise, partial volume and tissue variation, so it moves with the "
            "within-tissue SNRs as well as with contrast. Compare only with scans of the "
            "same sequence and field strength: the grey-white contrast is set by the "
            "sequence (inversion time, flip angle, echo time) far more than by image "
            "quality."),
        causes=(
            "Noise, motion (which blurs the boundary and widens both classes) and "
            "residual bias lower it. Grey-white contrast also changes over the lifespan: "
            "it is very different in infants and falls in older adults."),
        caveats=(
            "Not computed where the air is not real (defaced, skull-stripped, zero-filled). "
            "It mixes two kinds of spread (the raw air SD and the tissues' own), so read "
            "it as a relative measure within a protocol, not as a physical CNR."),
        mriqc="Same formula as MRIQC's cnr.",
        references=(_MRIQC,),
    ),
    "anat.cjv": Explanation(
        title="Coefficient of joint variation",
        short=("The spread of grey and white matter over the distance between them "
               "(Ganzetti 2016). Lower is better."),
        measures=(
            "How much the grey- and white-matter intensity distributions overlap: their "
            "combined spread relative to the gap between their centres. Large when the "
            "two tissues are hard to tell apart."),
        computed=(
            "(MAD_WM + MAD_GM) / |median WM - median GM| on the bias-corrected intensities "
            "at the image's resolution. Each MAD is the median absolute deviation scaled "
            "to an SD, over the voxels the class holds for sure (weight above half the "
            "class's largest); the medians are probability-weighted. It needs no air, so "
            "it is computed on defaced and skull-stripped images too."),
        reading=(
            "Lower is better. It works like the inverse of a contrast-to-noise ratio built "
            "from robust statistics: it rises with noise and blur and falls with stronger "
            "contrast. Compare with scans of the same sequence and field strength."),
        causes=(
            "Noise, motion (blur and ghosting widen both classes), residual intensity "
            "non-uniformity, and low grey-white contrast from the sequence or the "
            "participant's age."),
        caveats=(
            "The bias field is removed before it is measured, so it shows only the "
            "non-uniformity the correction left, not the raw field (see Intensity "
            "non-uniformity). Like every tissue measure, it depends on the segmentation."),
        mriqc="Same formula as MRIQC's cjv (MADs over the gap between medians).",
        references=(_GANZETTI,),
    ),
}

_ANAT_ARTEFACTS: dict[str, Explanation] = {
    "anat.efc": Explanation(
        title="Entropy focus criterion",
        short=("How evenly the image's energy is spread over its voxels (Atkinson 1997). "
               "Lower is better: ghosting and motion blur spread energy into the background."),
        measures=(
            "The Shannon entropy of the voxel intensities, normalised: a sharp image "
            "concentrates its energy in the head, while ghosting and blurring spread it "
            "into the air."),
        computed=(
            "Over every voxel except the zero fill, with x the intensities clipped at 0 "
            "and B = sqrt(sum x^2): E = sum (x/B) ln(x/B), divided by the value E takes "
            "when every voxel is equal. 0 means all the energy in one voxel, 1 an image of "
            "uniform intensity."),
        reading=(
            "Lower is better. Its value depends on the field of view and on the number of "
            "voxels, so compare only within one protocol, as the 'In this dataset' column "
            "does."),
        causes=("Motion (ghosts and blurring), Nyquist or other ghosting, wrap-around, and "
                "noise in the background."),
        caveats=(
            "Computed only on images whose air is real: a defaced, skull-stripped or "
            "zero-filled background changes the energy distribution for reasons that have "
            "nothing to do with the scan. Background noise raises it too, so a noisy but "
            "sharp image can read like a blurred one."),
        mriqc=(
            "Same formula as MRIQC's efc; on the images compared during development the "
            "two agreed to within about 0.002."),
        references=(_ATKINSON,),
    ),
    "anat.fber": Explanation(
        title="Foreground to background energy",
        short="The head's typical energy over the air's (Shehzad 2015). Higher is better.",
        measures=("How much brighter the head is than its background, in energy (squared "
                  "intensity)."),
        computed=(
            "The median of the squared intensities inside the head mask over the median "
            "of the squared intensities everywhere outside it, zero fill excluded. Unlike "
            "the other air measures it uses all the background, below the face as well as "
            "above it."),
        reading=(
            "Higher is better. It is large on a clean scan with a dark background and "
            "falls when signal leaks into the air. Compare within one protocol: the head "
            "mask, the field of view and the scanner's background handling all move it."),
        causes=("Ghosting, wrap-around, motion and noise in the air lower it, and so do flow "
                "artefacts that land outside the head."),
        caveats=(
            "Not computed when the air is not real (defaced, skull-stripped, zero-filled) "
            "or when the air's energy is zero. A scanner or tool that filters or scales "
            "the background raises it without any change in the head."),
        mriqc=(
            "Same formula as MRIQC's fber (medians of squared intensities). Where MRIQC "
            "returns -1 for an air of zero energy, BIDS Manager leaves the value empty."),
        references=(_SHEHZAD,),
    ),
    "anat.qi1": Explanation(
        title="Artefacts in the air (QI1)",
        short=("The share of the air above the face holding structured signal: ghosts, "
               "ringing, motion, wrap-around (Mortamet 2009). Lower is better."),
        measures=("How much of the background holds signal that is not noise, as a "
                  "percentage of the air measured."),
        computed=(
            "On the working grid (about 2 mm), an air voxel counts when its intensity is "
            "more than 10 times the air's robust spread (its median absolute deviation "
            "scaled to an SD) and it lies further from the head than a tenth of the "
            "largest distance in the air. Isolated voxels are then removed by a "
            "morphological opening. The value is what remains as a percentage of the air "
            "above the face."),
        reading=(
            "Lower is better; 0 is usual for a clean scan. Above 0.5 % by default "
            "(Settings > Quality control > Anatomical images) the check reports "
            "structured signal in the air, and the 'Artefacts in the air' layer shows "
            "where it is."),
        causes=(
            "Motion ghosts along the phase-encoding direction, ringing, wrap-around (part "
            "of the head folded into the far side of the field of view), and objects that "
            "give signal, such as a fiducial marker."),
        caveats=(
            "Only the air above the plane through the glabella and the inion is used: "
            "below it, folding and motion of the face, neck and shoulders are normal. Air "
            "close to the head is left out, so a ghost that overlaps the head itself is "
            "not counted. Not computed when the air is not real (defaced, skull-stripped, "
            "zero-filled)."),
        mriqc=(
            "MRIQC reports qi_1 with the same intent, but BIDS Manager's development notes "
            "record that MRIQC's artefact mask is always empty, so its qi_1 reads 0. This "
            "value follows Mortamet's definition as MRIQC meant it."),
        references=(_MORTAMET,),
    ),
    "anat.qi2": Explanation(
        title="Noise fit (QI2)",
        short=("How poorly a chi-squared distribution fits the upper tail of the air's "
               "intensities (Mortamet 2009). Lower is better."),
        measures=(
            "Whether the background holds only noise. Pure noise in a magnitude image has "
            "a smooth distribution of a known family; structured signal in the air adds an "
            "upper tail that no such fit follows."),
        computed=(
            "The air's positive intensities at the image's resolution (up to 100,000 "
            "sampled) are scaled so that their 99th percentile is 100, their density is "
            "estimated with a Gaussian kernel, and a chi-squared distribution is fitted to "
            "them. The value is the mean absolute difference between the density and the "
            "fit over the upper tail, beyond the last point where the density is above "
            "half its peak."),
        reading=(
            "Lower is better: small values mean the air looks like noise, larger ones that "
            "something else is there. Read it with QI1 and the 'Artefacts in the air' "
            "layer, which show where."),
        causes=("Ghosts, ringing, wrap-around and motion artefacts in the air, and a "
                "background the scanner filtered or rescaled, which changes the noise's shape."),
        caveats=(
            "Not computed with fewer than 1000 air voxels or when the air is not real. "
            "Multi-channel coils and parallel imaging change the noise distribution "
            "itself, so the fit can be poor without any artefact: compare within one "
            "protocol."),
        mriqc=("Same algorithm as MRIQC's qi_2; BIDS Manager estimates the density in numpy "
               "rather than with a library kernel estimator."),
        references=(_MORTAMET,),
    ),
    "anat.ghost_ratio": Explanation(
        title="Ghost to signal ratio",
        short=("The extra signal where a ghost of the head would fall along the "
               "phase-encoding direction, over the head's mean. Lower is better."),
        measures=(
            "Ghosting: faint copies of the head displaced along the phase-encoding "
            "direction. It compares the air where a copy shifted by half the field of view "
            "would land with the rest of the air."),
        computed=(
            "On the working grid, the head mask is shifted by half the field of view along "
            "the phase-encoding axis (PhaseEncodingDirection, else "
            "InPlanePhaseEncodingDirectionDICOM, from the sidecar) and intersected with the "
            "measured air. The value is (mean of that ghost region - mean of the rest of "
            "the air) / mean of the head."),
        reading=(
            "Lower is better. Near 0 the ghost region is no brighter than the rest of the "
            "air, and a slightly negative value is noise; a clear positive value means "
            "signal repeats at the half-field position."),
        causes=("Nyquist (N/2) ghosting of echo-planar readouts, and motion or pulsation "
                "ghosts that reach the half-field position."),
        caveats=(
            "Anatomical sequences are rarely echo-planar, and their motion ghosts repeat at "
            "spacings set by the motion, so only the part that reaches the half-field "
            "position counts; the 'Artefacts in the air' layer shows ghosts wherever they "
            "are. Not computed without a phase-encoding direction in the sidecar, with too "
            "little air at the ghost position, or when the air is not real."),
        mriqc=("No anatomical equivalent in MRIQC; its functional report has a ghost-to-"
               "signal ratio of the same idea (gsr_x, gsr_y)."),
    ),
}

_ANAT_TISSUES: dict[str, Explanation] = {
    "anat.wm2max": Explanation(
        title="White matter to maximum",
        short=("White matter's median over the top of the image's intensity range. Neither "
               "direction is simply better: very low values mean a few very bright voxels "
               "take up the range."),
        measures=(
            "Where white matter sits in the intensity range: whether the range is used by "
            "the brain or taken up by a long tail of bright voxels (vessels, fat, an "
            "artefact)."),
        computed=(
            "White matter's bias-corrected, probability-weighted median over the 99.9th "
            "percentile of all non-zero voxels of the stored image."),
        reading=(
            "Neither: read it against images of the same contrast and protocol. MRIQC's "
            "documentation gives about 0.6 to 0.8 as usual for a T1w, where fat in the "
            "scalp and marrow is brighter than white matter. Much lower values mean a long "
            "bright tail; values near or above 1 mean nothing is brighter than white "
            "matter, as in a skull-stripped or fat-suppressed image. On a T2w white matter "
            "is dark and the value is low by nature."),
        causes=(
            "Bright vessels (especially with a contrast agent), fat, a bright artefact or "
            "a hot spot near a coil element stretch the top of the range; intensity "
            "rescaling at conversion can change it too."),
        caveats=(
            "The numerator is bias-corrected and the denominator comes from the "
            "uncorrected image, so a strong bias field moves the ratio. Needs the tissue "
            "classes."),
        mriqc=(
            "Same idea as MRIQC's wm2max (white matter's median over a high percentile of "
            "the image); values differ through the segmentation and the exact ceiling."),
    ),
    "anat.inu_range": Explanation(
        title="Intensity non-uniformity",
        short=("How much the intensity bias field varies over the brain: its 95th minus "
               "5th percentile, the field normalised to 1. Lower is better; 0 is uniform."),
        measures=(
            "The slow, smooth change of brightness across the image that comes from the "
            "coil's sensitivity and the RF field rather than from the tissue."),
        computed=(
            "The bias field is fitted with the tissue classes on the working grid (about "
            "2 mm), N3's idea: a smooth fit, a third-order polynomial plus a Gaussian "
            "smoothing (sigma about 10 mm on the default grid) of what the polynomial "
            "leaves, to the part of the log image the classes do not explain. It is "
            "turned into a multiplier inside the brain and normalised to median 1; the "
            "value is its 95th percentile minus its 5th, so 0.2 means those two "
            "percentiles of the field lie 20 % of the median apart."),
        reading=(
            "Lower is better, within a protocol. Large values are common with "
            "multi-channel head coils and at higher field strengths when the scanner did "
            "not normalise the intensities, and smaller when it did."),
        causes=(
            "Receive-coil sensitivity (strong near the elements of a multi-channel array), "
            "transmit (B1) inhomogeneity, which grows with field strength, and the head's "
            "position in the coil."),
        caveats=(
            "The smooth fit can also absorb real, slowly varying tissue differences (deep "
            "grey nuclei against the cortex, for example), so part of the value can be "
            "anatomy. The tissue measures are taken after this field is removed, so a "
            "large value does not by itself lower the SNRs: it says how much the "
            "correction had to do."),
        mriqc=("Not comparable with MRIQC's inu_range or inu_med, which read N4's field on "
               "its own scale."),
        references=(_SLED,),
    ),
    "anat.fraction_csf": Explanation(
        title="CSF fraction",
        short=("The share of the brain mask classified as CSF. Neither direction: a check "
               "on the segmentation and the anatomy."),
        measures=("CSF's share of the segmented brain volume; the three fractions add up "
                  "to 1."),
        computed=(
            "The sum of the CSF probabilities over the brain on the working grid, over the "
            "sum of all three classes. With the tissue network, CSF is everything inside "
            "the brain mask that the network does not label grey or white matter."),
        reading=(
            "Neither: compare with the dataset's other images of the same contrast. It "
            "rises with age and atrophy (larger ventricles and sulci). An outlier more "
            "often means a brain mask or a segmentation that failed than an unusual brain: "
            "check the 'Tissue classes' layer."),
        causes=(
            "How far the brain mask reaches (extra-cerebral CSF, dura and vessels inside "
            "it count as CSF), the segmentation method, partial volume, and anatomy: age, "
            "atrophy, enlarged ventricles."),
        caveats=(
            "Not a clinical volume: the brain mask is a quality-check mask, not an "
            "intracranial volume. Values from the networks and from the fast fallback "
            "differ."),
        mriqc="Same idea as MRIQC's icvs_csf (each class's share of the total).",
    ),
    "anat.fraction_gm": Explanation(
        title="Grey matter fraction",
        short=("The share of the brain mask classified as grey matter. Neither direction: "
               "a check on the segmentation and the anatomy."),
        measures="Grey matter's share of the segmented brain volume.",
        computed=(
            "The sum of the grey-matter probabilities over the brain on the working grid, "
            "over the sum of all three classes."),
        reading=(
            "Neither: compare with the dataset's other images of the same contrast. It "
            "falls with age as the cortex thins, and depends strongly on where the "
            "segmentation puts the border with white matter. An outlier more often means a "
            "segmentation problem than unusual anatomy."),
        causes=(
            "The segmentation method, grey-white contrast (low contrast moves the border), "
            "partial volume, blur, and anatomy."),
        caveats=(
            "Not a clinical volume. Values from the networks and from the fast fallback "
            "differ."),
        mriqc="Same idea as MRIQC's icvs_gm.",
    ),
    "anat.fraction_wm": Explanation(
        title="White matter fraction",
        short=("The share of the brain mask classified as white matter. Neither direction: "
               "a check on the segmentation and the anatomy."),
        measures="White matter's share of the segmented brain volume.",
        computed=(
            "The sum of the white-matter probabilities over the brain on the working grid, "
            "over the sum of all three classes."),
        reading=(
            "Neither: compare with the dataset's other images of the same contrast. It "
            "changes over the lifespan (white matter grows through childhood and "
            "adolescence and shrinks in older age). An outlier more often means a "
            "segmentation problem than unusual anatomy."),
        causes=("The segmentation method, grey-white contrast, partial volume, lesions, and "
                "anatomy."),
        caveats=(
            "Not a clinical volume. White-matter lesions can be classed as grey matter or "
            "CSF and lower it. Values from the networks and from the fast fallback differ."),
        mriqc="Same idea as MRIQC's icvs_wm.",
    ),
    "anat.partial_volume": Explanation(
        title="Partial volume",
        short=("The share of brain voxels that no tissue class holds with a probability of "
               "0.9 or more. Lower is better."),
        measures=(
            "How many voxels are mixtures of tissues or undecided: voxels at the borders "
            "between tissues, and voxels that noise or blur leaves uncertain."),
        computed=(
            "At the image's own resolution, on the sampled brain voxels, with the class "
            "probabilities of the estimate made again at that resolution (the intensity's "
            "likelihood under each class times a prior from the tissue network or the "
            "working-grid classes). A voxel counts when its largest probability is below "
            "0.9 by default (Settings > Quality control > Anatomical images)."),
        reading=(
            "Lower is better, against scans of the same voxel size and contrast. A rise "
            "shows blur, motion, thicker voxels or more noise, all of which leave more "
            "voxels undecided."),
        causes=("Thick or large voxels, motion blur, reconstruction filtering or resampling, "
                "noise, and low grey-white contrast."),
        caveats=(
            "Partly anatomy: the folded cortex always leaves many border voxels, and a "
            "brain with wider sulci has more. It depends on the segmentation's priors, so "
            "values from the networks and from the fast fallback differ."),
        mriqc="Not MRIQC's rpve, which is not a fraction and is not comparable.",
    ),
    "anat.template_overlap_gm": Explanation(
        title="Overlap with the template (grey matter)",
        short=("How well the grey-matter map matches the template's after an affine "
               "registration (fuzzy Jaccard index). Higher is better."),
        measures=(
            "Agreement between this image's grey-matter map and the template's (MNI 2009c) "
            "brought onto the image."),
        computed=(
            "The template's tissue maps are carried onto the working grid by the affine "
            "registration (niimath -allineate, or BIDS Manager's own fit). The value is "
            "sum(min(p, q)) / sum(max(p, q)) over the image, p the image's grey-matter "
            "probabilities and q the template's."),
        reading=(
            "Higher is better, against the dataset. Values well below 1 are normal, since "
            "an affine registration cannot match individual folding. A clear outlier "
            "points at a failed registration or segmentation, or at anatomy far from the "
            "template."),
        causes=(
            "A registration that failed or is uncertain (reported as a finding), a wrong "
            "segmentation, a cut field of view, large ventricles, lesions, or a brain shape "
            "far from the adult template."),
        caveats=(
            "It measures the registration and the segmentation together and is not an "
            "image-quality measure by itself. Not computed when the template is not "
            "registered (slabs thinner than 70 mm by default, Settings > Quality control > "
            "Anatomical images)."),
        mriqc=("MRIQC's tpm_overlap_gm is the same fuzzy overlap; values differ through the "
               "registration and the segmentation."),
        references=(_CRUM,),
    ),
    "anat.template_overlap_wm": Explanation(
        title="Overlap with the template (white matter)",
        short=("How well the white-matter map matches the template's after an affine "
               "registration (fuzzy Jaccard index). Higher is better."),
        measures=(
            "Agreement between this image's white-matter map and the template's (MNI 2009c) "
            "brought onto the image."),
        computed=(
            "As for grey matter: sum(min(p, q)) / sum(max(p, q)) over the working grid, p "
            "the image's white-matter probabilities and q the template's, carried by the "
            "affine registration."),
        reading=(
            "Higher is better, against the dataset. White matter is a larger, smoother "
            "structure than the cortex, so it usually overlaps better than grey matter; a "
            "clear outlier points at a failed registration or segmentation."),
        causes=(
            "A failed or uncertain registration, a wrong segmentation, a cut field of view, "
            "enlarged ventricles, lesions, or anatomy far from the adult template."),
        caveats=(
            "It measures the registration and the segmentation together. Not computed when "
            "the template is not registered (slabs thinner than 70 mm by default)."),
        mriqc=("MRIQC's tpm_overlap_wm is the same fuzzy overlap; values differ through the "
               "registration and the segmentation."),
        references=(_CRUM,),
    ),
}


def _fwhm_axis(axis: str) -> Explanation:
    """The smoothness along one voxel axis (``axis``: "x", "y" or "z")."""
    return Explanation(
        title=f"Smoothness along {axis}",
        short=(f"The full width at half maximum of the image's spatial correlation along "
               f"{axis}, in mm (Forman 1995). Lower means sharper, but noise lowers it too."),
        measures=(
            "How smooth the image is along one axis: how strongly neighbouring voxels "
            "resemble each other, expressed as the width of an equivalent Gaussian blur."),
        computed=(
            f"Inside the brain mask, at the image's own resolution, on the stored "
            f"intensities: rho = 1 - var(differences between neighbours along {axis}) / "
            f"(2 var(intensities)), then FWHM = voxel size x sqrt(-2 ln 2 / ln rho). "
            f"The axis is the image's own voxel axis, which is left-right, back-front or "
            f"down-up only when the image is stored that way."),
        reading=(
            "Lower is sharper. It is in millimetres, so larger voxels give larger values: "
            "compare only within a protocol. The brain's own structure (grey against white "
            "matter) makes neighbours alike, so the value is several millimetres even for "
            "a sharp scan and is not the scanner's blur alone."),
        causes=(
            "Motion blur, reconstruction filtering, zero-filling interpolation and partial "
            "Fourier, any resampling (reorientation, registration, or a tool that rewrote "
            "the image), and larger voxels raise it."),
        caveats=(
            "Noise makes neighbours less alike and lowers it, so a noisy image reads "
            "sharper: read it beside the SNRs. A difference between the axes is expected "
            "when the voxels or the resolution differ along one of them."),
        mriqc=(f"MRIQC reports fwhm_{axis} (AFNI's 3dFWHMx) in voxels; BIDS Manager reports "
               f"millimetres from Forman's estimator."),
        references=(_FORMAN,),
    )


_ANAT_COVERAGE: dict[str, Explanation] = {
    "anat.fwhm_x": _fwhm_axis("x"),
    "anat.fwhm_y": _fwhm_axis("y"),
    "anat.fwhm_z": _fwhm_axis("z"),
    "anat.fwhm_avg": replace(
        _fwhm_axis("x"),
        title="Smoothness (mean)",
        short=("The mean of the smoothness along the three voxel axes, in mm. Lower means "
               "sharper, but noise lowers it too."),
        measures=(
            "How smooth the image is overall: how strongly neighbouring voxels resemble "
            "each other, as the width of an equivalent Gaussian blur averaged over the "
            "three axes."),
        computed=(
            "The mean of the three per-axis values (any that could not be computed is "
            "left out). Each is Forman's estimate inside the brain mask at the image's own "
            "resolution: rho = 1 - var(neighbour differences) / (2 var(intensities)), then "
            "FWHM = voxel size x sqrt(-2 ln 2 / ln rho)."),
        mriqc=("MRIQC reports fwhm_avg (AFNI's 3dFWHMx) in voxels; BIDS Manager reports "
               "millimetres from Forman's estimator."),
    ),
    "anat.fov_cut": Explanation(
        title="Brain outside the field of view",
        short=("The share of the template's brain that falls outside the image after "
               "registration: a cut top of the head or cerebellum. Lower is better."),
        measures="Whether the field of view covers the whole brain.",
        computed=(
            "Every brain voxel of the template is carried through the registration into "
            "the image's voxel grid; the value is the percentage that lands outside it. "
            "The finding names the sides (top, bottom, front, back, left, right)."),
        reading=(
            "Lower is better; 0 is a complete brain. Above 0.5 % by default the check "
            "warns and above 3 % it reports an error (Settings > Quality control > "
            "Anatomical images). A slab thinner than 120 mm by default covers part of the "
            "brain on purpose, and the same value is then reported as coverage."),
        causes=("A field of view placed too low or too high, a small field of view, a "
                "large head, or a tilted head."),
        caveats=(
            "Only as good as the registration: an uncertain registration (reported as a "
            "finding) makes the value uncertain too. It counts the template's brain, not "
            "the participant's own, so it depends on how well the fit holds at the edges. "
            "Not computed for slabs thinner than 70 mm by default, where no template is "
            "registered."),
    ),
    "anat.saturation": Explanation(
        title="Saturated voxels",
        short=("The share of the head at the image's maximum value: bright tissue clipped "
               "at the top of the stored range. Lower is better."),
        measures=(
            "Whether intensities were clipped: many voxels sharing exactly the highest "
            "value mean that the scanner or a conversion ran out of range."),
        computed=("At the image's resolution, inside the head mask: the percentage of voxels "
                  "whose value equals the head's maximum."),
        reading=(
            "Lower is better; in an unclipped image it is a tiny fraction of a percent. "
            "Above 0.5 % by default (Settings > Quality control > Anatomical images) the "
            "check reports saturated voxels."),
        causes=(
            "A receiver or reconstruction scaling that overflowed the stored range, a "
            "conversion that rescaled and clipped, or very bright fat or vessels at the "
            "limit of the data type."),
        caveats=(
            "Detail above the clipped value is lost and cannot be recovered. An image with "
            "very few distinct values (heavily quantised, or a mask) also has many voxels "
            "at its maximum."),
    ),
}


# ---------------------------------------------------------------------------
# Diffusion: the groups of the measures table
# ---------------------------------------------------------------------------

_DWI_GROUPS: dict[str, Explanation] = {
    "group.dwi.gradients": Explanation(
        title="Gradient table",
        short=("Whether the .bval and .bvec describe the series and cover the directions "
               "evenly. A wrong table ruins every analysis and no image measure shows it."),
        measures=(
            "The table itself, before any image: present, one entry per volume, unit "
            "vectors, at least one b=0, the shells, repeated directions, how evenly each "
            "shell covers the sphere, and (experimental) whether the vectors look flipped "
            "or swapped."),
        computed=(
            "Volumes with a b-value of 50 or below by default are b=0 (Settings > Quality "
            "control > Diffusion); the others are grouped into shells rounded to the "
            "nearest 50. Errors: a missing table, a count that does not match the volumes, "
            "a .bvec that is not three rows, a diffusion volume with no direction. "
            "Warnings: vectors whose length is off 1 by more than 0.01, a shell with fewer "
            "than six directions; a note for pairs within 2 degrees by default. The flip "
            "check fits the tensor with the stored table, with each axis reversed and with "
            "each reordering of the axes, and reports another table when the principal "
            "directions are more continuous with it by 2 % by default."),
        reading=(
            "Read the findings first: an error here makes the image measures meaningless. "
            "The largest gap per shell is small when the directions are spread evenly; a "
            "large gap leaves part of the sphere unsampled, which biases estimates of "
            "fibres pointing that way. A reversed or swapped table leaves FA, MD and the "
            "neighbouring-direction correlation unchanged, which is why it needs its own "
            "check; confirm a reported flip by eye (fibre directions on a colour-coded FA "
            "image) before changing the .bvec."),
        causes=(
            "Conversion errors, volumes dropped or reordered after acquisition, a scan "
            "stopped early, and the convention mismatch between image axes and scanner "
            "axes."),
        caveats=(
            "The flip check is experimental and needs a tensor fit: it compares candidate "
            "tables, it does not prove one right."),
        references=(_JEURISSEN, _JONES99),
    ),
    "group.dwi.motion": Explanation(
        title="Motion",
        short=("How the head moved during the series: from the b=0 volumes, the reliable "
               "record, and from every volume registered to what the tensor predicts for it."),
        measures=(
            "Displacement across the b=0 volumes, the largest step between them, volumes "
            "far out of place, the mean framewise displacement, eddy-current shifts, the "
            "largest rotation, and the drift of the b=0 signal."),
        computed=(
            "Every volume is registered rigidly on a grid of about 3 mm: each b=0 to the "
            "median of the b=0 volumes, each diffusion-weighted volume to what the tensor "
            "predicts for its direction and b-value (to its shell's mean when no tensor "
            "can be fitted). Positions are taken relative to the first b=0. In each shell "
            "of at least 10 volumes, the part of the translations linear in the gradient's "
            "components is taken as eddy-current shift and removed from the motion when it "
            "explains at least 30 % of them by default. Displacement adds the three "
            "translations and the three rotations as arcs on a 50 mm sphere by default "
            "(both Settings > Quality control > Diffusion)."),
        reading=(
            "The b=0 measures (motion across the b=0 volumes, the largest step between "
            "them) are the most trustworthy: high signal, one contrast. Measures over every "
            "volume (the mean framewise displacement) include the registration's noise on "
            "single low-signal volumes, so read them against scans of the same protocol, "
            "not against a BOLD threshold. Volumes far out of place point at a movement "
            "during one volume, and the largest rotation says whether the b-vectors should "
            "be rotated with the motion correction. Drift is a scanner effect, not motion, "
            "but it is read from the same b=0 volumes."),
        causes=("Head motion, eddy currents left in, and the registration's noise on "
                "low-signal volumes."),
        caveats=(
            "A screening registration, not the correction an analysis uses (FSL eddy or "
            "similar). A volume registered beyond 30 mm or 15 degrees by default (Settings "
            "> Quality control > Diffusion) is reported as not placed and counted as "
            "unmoved."),
        references=(_POWER12, _LEEMANS),
    ),
    "group.dwi.noise": Explanation(
        title="Noise and contrast",
        short=("How large the diffusion signal is against the noise: two noise estimates, "
               "the SNR in the corpus callosum per shell, and how much of the highest shell "
               "sits at the noise floor."),
        measures=(
            "The noise SD, by MP-PCA and from repeated b=0 volumes; the corpus callosum's "
            "signal over that noise at b=0 and, per shell, along and across its fibres; "
            "and the share of tissue signal at the highest shell too close to the noise to "
            "trust."),
        computed=(
            "MP-PCA works on sampled patches of the shell with the most volumes; the b=0 "
            "estimate is the spread across repeated b=0 volumes. MP-PCA is used for the "
            "SNRs when it is available, because the b=0 repeats also hold motion. The "
            "corpus callosum comes from the tensor: FA above 0.4, principal direction "
            "within about 32 degrees of left-right, within 10 mm of the brain's midline, "
            "the largest connected part."),
        reading=(
            "The callosal SNR is the diffusion counterpart of an anatomical image's "
            "white-matter SNR: one well-defined tissue, signal over noise. Along the fibres "
            "(gradient left-right) the signal is attenuated most, the worst case; across "
            "them least. At the highest shell, the along value says whether the signal "
            "still stands above the noise, and the noise-floor measure how much of the "
            "tissue signal is too small to trust in magnitude data. Compare all of these "
            "within one protocol: voxel size, b-values, coil, acceleration and the number "
            "of averages change them."),
        causes=(
            "Small voxels, high b-values, long echo times, acceleration, few averages, and "
            "the low sensitivity of a multi-channel coil in the centre of the head."),
        caveats=(
            "The noise estimates assume noise that is the same everywhere and independent "
            "between voxels; parallel imaging, partial Fourier and any interpolation break "
            "both assumptions. The callosal measures need a tensor fit and at least 5 "
            "callosal voxels."),
        references=(_VERAART_NI, _GUDBJARTSSON),
    ),
    "group.dwi.artefacts": Explanation(
        title="Artefacts",
        short=("Artefacts found by comparing each slice and voxel with what the tensor "
               "predicts (dropout, interleave, spikes), plus EFC and FBER per shell."),
        measures=(
            "Slices that lost signal in one volume, volumes whose odd and even slices "
            "disagree, single voxels far above their prediction, and the ghosting and "
            "background of each shell's mean image."),
        computed=(
            "A tensor fitted on the b=0 volumes and b-values up to 1500 by default "
            "predicts every volume's signal. For each slice of each volume, the mean "
            "difference between the observed log signal and that prediction is compared "
            "with the same slice in the shell's other volumes as a robust z. A slice with "
            "fewer brain voxels than 40, or than 25 % of a typical slice by default, is not "
            "judged (Settings > Quality control > Diffusion). Spikes compare each voxel "
            "with itself across the shell."),
        reading=(
            "Signal dropout is the commonest diffusion artefact, and MRIQC does not look "
            "for it: a slice far below its prediction in one volume. The 'Slice signal' "
            "plot shows every slice of every volume, and a click goes there. Interleave "
            "artefacts and spikes are rarer and point at motion between slice passes and "
            "at hardware. EFC and FBER summarise ghosting and the background per shell, as "
            "for anatomical images."),
        causes=("Motion during the diffusion encoding, cardiac pulsation, table vibration, "
                "RF spikes, and Nyquist ghosting."),
        caveats=(
            "Without a tensor (fewer than six directions at b-values up to the fitting "
            "limit, or no b=0) the per-slice and spike measures are not computed. EFC and "
            "FBER need air, and the diffusion check treats an image as defaced only when "
            "the sidecar records it."),
        references=(_ANDERSSON_OLR,),
    ),
    "group.dwi.diffusion": Explanation(
        title="Diffusion",
        short=("Sanity checks of the diffusion signal: the tensor's median FA and MD, how "
               "often the tensor is not physical, and whether neighbouring directions agree."),
        measures=(
            "FA and MD summarise the tensor over the brain; voxels with a negative "
            "eigenvalue count unphysical fits; the neighbouring-direction correlation says "
            "whether volumes with similar directions look alike."),
        computed=(
            "One tensor per brain voxel, by weighted least squares on the b=0 volumes and "
            "b-values up to 1500 by default (Settings > Quality control > Diffusion). The "
            "brain is mindgrab's on the median b=0, or a median filter and Otsu's "
            "threshold, as Settings > Quality control > Methods chooses."),
        reading=(
            "FA and MD have no better direction: they check the data and the table. An MD "
            "far from tissue values for a whole protocol points at the b-values; an FA or "
            "MD outlier for one scan at its brain mask, its noise or its motion. Negative "
            "eigenvalues and a low neighbouring-direction correlation both rise with noise, "
            "motion and artefacts, and the per-volume plots show where. Reversing or "
            "swapping axes of the whole table changes none of these, because it reflects "
            "or rotates the tensor without changing its eigenvalues; the flip check under "
            "Gradient table is the test for that."),
        causes=("Noise, motion, artefacts, a brain mask that is too loose or too tight, and "
                "wrong b-values."),
        caveats=(
            "They need a tensor: at least six directions at b-values up to the fitting "
            "limit, and a b=0 volume. The medians are over the whole brain mask, so they "
            "mix white matter, grey matter and CSF."),
        references=(_BASSER94, _YEH),
    ),
}


# ---------------------------------------------------------------------------
# Diffusion: the measures
# ---------------------------------------------------------------------------

_DWI_GRADIENTS: dict[str, Explanation] = {
    "dwi.coverage_gap_b*": Explanation(
        title="Largest gap between directions",
        short=("The radius, in degrees, of the largest region of the sphere with no "
               "gradient direction of this shell. Lower is better."),
        measures=(
            "How evenly the shell's directions cover the sphere, either sign counting (a "
            "direction and its opposite measure the same thing)."),
        computed=(
            "4000 random points on the sphere are each matched with the shell's nearest "
            "direction, either sign; the value is the largest of those angles, the radius "
            "of the biggest empty cap."),
        reading=(
            "Lower is better, but the number of directions sets a floor: few directions "
            "cannot cover the sphere closely however they are placed. Compare with shells "
            "of the same number of directions; a value far above a well-spread scheme of "
            "the same count means clustered directions or missing volumes."),
        causes=("Few directions, a poorly designed scheme, and volumes dropped at conversion "
                "or by a scan stopped early."),
        caveats=(
            "A property of the table, not of the image: it says nothing about whether the "
            "images match the table. It is estimated from random points, so it can be a "
            "little below the true largest gap."),
        references=(_JONES99,),
    ),
}

_DWI_MOTION: dict[str, Explanation] = {
    "dwi.motion_b0": Explanation(
        title="Head motion across the b=0 volumes",
        short=("The largest displacement of any b=0 volume from the first, in mm. Lower is "
               "better."),
        measures="How far the head moved over the scan, as the b=0 volumes record it.",
        computed=(
            "Each b=0 volume is registered rigidly to the median b=0 image. Its position "
            "relative to the first b=0 is summed as the three absolute translations plus "
            "the three absolute rotations as arcs on a 50 mm sphere by default (Settings > "
            "Quality control > Diffusion); the value is the largest over the b=0 volumes."),
        reading=(
            "Lower is better. Above 1 mm by default (same section) the check reports head "
            "motion, as an error from three times that. With b=0 volumes spread through "
            "the series it tracks motion across the whole scan; with all of them at the "
            "start it says little about the rest."),
        causes="Head motion, including slow settling into the pillow over a long scan.",
        caveats=(
            "With a single b=0 volume it reads 0, with nothing to compare. It is a "
            "displacement from the first b=0, not a frame-to-frame change, so slow drift "
            "and a single jump read alike: see the 'Displacement' plot."),
    ),
    "dwi.fd_b0_max": Explanation(
        title="Largest step between b=0 volumes",
        short=("The largest change of head position between consecutive b=0 volumes, in "
               "mm. Lower is better."),
        measures="The biggest movement between one b=0 volume and the next.",
        computed=(
            "Framewise displacement (Power 2012) between each b=0 volume and the next b=0, "
            "from the same registration: absolute changes of the three translations plus "
            "absolute changes of the three rotations as arcs on a 50 mm sphere by default "
            "(Settings > Quality control > Diffusion); the largest is reported."),
        reading=(
            "Lower is better, against the same protocol. Consecutive b=0 volumes can be "
            "many volumes apart, so this is the movement over that interval, not between "
            "two neighbouring volumes."),
        causes="Head motion between the b=0 volumes.",
        caveats=("Needs two b=0 volumes or more. It misses a movement that returns to the "
                 "same position between two b=0 volumes."),
        references=(_POWER12,),
    ),
    "dwi.volumes_far": Explanation(
        title="Volumes far out of place",
        short=("The number of volumes displaced far from the first b=0 and far from the "
               "rest of their shell. Lower is better; 0 is usual."),
        measures="Volumes during which the head probably moved.",
        computed=(
            "A volume counts when its displacement from the first b=0 is above 2 mm by "
            "default (Settings > Quality control > Diffusion) and its robust z among the "
            "volumes of its own shell (or among the b=0 volumes) is above 5. A shell with "
            "fewer than four volumes is not judged."),
        reading=(
            "Lower is better. Each such volume probably moved during its own acquisition; "
            "the 'Displacement' plot marks them. When they are more than 5 % of the "
            "volumes the finding is an error rather than a warning."),
        causes=("A sudden movement, a cough or a swallow during that volume; a registration "
                "that failed on a low-signal volume can look the same."),
        caveats=(
            "Each volume is judged within its shell, so a scan in which every volume moved "
            "by the same amount shows none far out of place: read the motion across the "
            "b=0 volumes too."),
    ),
    "dwi.fd_mean": Explanation(
        title="Mean framewise displacement",
        short=("The mean framewise displacement over the whole series, in mm. Lower is "
               "better, against scans of the same protocol."),
        measures="The average volume-to-volume movement of the head over the series.",
        computed=(
            "Framewise displacement (Power 2012) between every pair of consecutive "
            "volumes, after the eddy-current part is removed, averaged. Each "
            "diffusion-weighted volume is registered to what the tensor predicts for its "
            "direction, each b=0 to the b=0 reference."),
        reading=(
            "Lower is better. It includes the registration's own noise on single "
            "low-signal volumes, larger at high b-values, so it reads higher than a BOLD "
            "run's FD for the same head motion: compare with other diffusion scans of the "
            "same protocol, not with a BOLD threshold such as 0.5 mm."),
        causes="Head motion, and registration noise that grows as the signal falls.",
        caveats=(
            "Not the motion an analysis would correct (FSL eddy or similar). A shell with "
            "few volumes or a poor tensor fit raises the registration noise."),
        mriqc=("Shown beside MRIQC's fd_mean; the two come from different registrations, so "
               "compare within one tool."),
        references=(_POWER12,),
    ),
    "dwi.eddy_shift": Explanation(
        title="Eddy-current shift",
        short=("The largest shift of a diffusion-weighted volume explained by its gradient "
               "direction, in mm: eddy currents. Lower is better."),
        measures=(
            "Eddy currents from the diffusion gradients shift (and also stretch and shear) "
            "each diffusion-weighted image, mostly along the phase-encoding axis, by an "
            "amount that depends on the gradient's direction. A rigid registration reads "
            "the shift as head motion."),
        computed=(
            "In each shell of at least 10 volumes, the translations are regressed on the "
            "gradient's three components; when the gradient explains at least 30 % of "
            "their variance by default (Settings > Quality control > Diffusion), the "
            "explained part is taken as eddy current and removed from the motion. The "
            "value is the largest absolute component of that part over every volume, and "
            "0 when no shell passed."),
        reading=(
            "Lower is better. A shift of a voxel or more means distortion correction (FSL "
            "eddy or similar) is needed; that is routine for diffusion data and not a "
            "fault of the scan."),
        causes=("Strong, fast-switching diffusion gradients, high b-values, and the "
                "scanner's eddy-current compensation."),
        caveats=(
            "Only the shift is measured, not the stretch and shear eddy currents also "
            "cause. A shell with few volumes, or head motion that happens to follow the "
            "gradient direction, can be misread."),
        references=(_ANDERSSON_EDDY,),
    ),
    "dwi.rotation_max": Explanation(
        title="Largest rotation of the gradients",
        short=("The largest rotation of the head against the first b=0, in degrees. Lower "
               "is better: the gradient directions turn with the head."),
        measures="How far the head turned, and so how far every gradient direction turned "
                 "relative to the head.",
        computed=(
            "The angle of each volume's rotation (the length of its rotation vector) "
            "relative to the first b=0, from the same registration; the largest is "
            "reported."),
        reading=(
            "Lower is better. Above 3 degrees the check adds a note: when motion is "
            "corrected, the b-vectors should be rotated with it, or the directions are "
            "wrong by up to that angle."),
        causes="Head rotation, most often nodding.",
        caveats="From a screening registration: small values are within its noise.",
        references=(_LEEMANS,),
    ),
    "dwi.drift": Explanation(
        title="Signal drift over the scan",
        short=("The change of the b=0 signal from the first volume to the last, in percent. "
               "Closer to zero is better; the sign says which way."),
        measures=(
            "A slow change of the scanner's signal over the series, usually from heating. "
            "Uncorrected, it biases every diffusion measure, because diffusion weighting "
            "is read as signal lost against the b=0."),
        computed=(
            "The median brain signal of each b=0 volume over the first's; a straight line "
            "is fitted to the log of that ratio against the volume index, and the value is "
            "the change the fit gives from the first volume to the last: (exp(slope x "
            "(n - 1)) - 1) x 100."),
        reading=(
            "Closer to zero is better. Above 5 % by default (Settings > Quality control > "
            "Diffusion) the check warns: correct the drift before fitting. The 'b=0 "
            "signal' plot shows the points the fit went through."),
        causes="Gradient and shim heating over a long, gradient-intensive scan.",
        caveats=(
            "Needs two b=0 volumes or more, and is only as good as their spread through "
            "the series: with every b=0 at the start, the fit is extrapolated over the "
            "rest. Drift is often not exponential, and motion also changes the median "
            "brain signal."),
        references=(_VOS,),
    ),
}

_DWI_DIFFUSION: dict[str, Explanation] = {
    "dwi.fa_median": Explanation(
        title="Median FA in the brain",
        short=("The median fractional anisotropy of the tensor over the brain mask. "
               "Neither direction: a sanity value to compare across the dataset."),
        measures=(
            "How directional diffusion is, from 0 (the same in every direction) to 1 "
            "(along one direction only), summarised over the brain."),
        computed=(
            "A tensor is fitted in every brain voxel by weighted least squares (ordinary "
            "least squares on the log signal, then one step weighted by the squared "
            "prediction) on the b=0 volumes and b-values up to 1500 by default (Settings > "
            "Quality control > Diffusion). Negative eigenvalues are set to 0 before FA is "
            "computed; the value is the median over the brain mask."),
        reading=(
            "Neither: compare with the dataset's other diffusion scans of the same "
            "protocol. The median mixes white matter, grey matter and CSF, so it moves "
            "with the brain mask as much as with the tissue; an outlier points at the "
            "mask, the b-values, heavy noise or motion."),
        causes=(
            "Noise raises FA in tissue of low anisotropy; motion and artefacts disturb the "
            "fit; a looser or tighter brain mask changes the mix of tissues; age and "
            "pathology change the real anisotropy."),
        caveats=(
            "A reversed or swapped b-vector table leaves FA unchanged, so this cannot "
            "detect one: see the flip check under Gradient table."),
        references=(_BASSER96,),
    ),
    "dwi.md_median": Explanation(
        title="Median MD in the brain",
        short=("The median mean diffusivity of the tensor over the brain mask, in um^2/ms. "
               "Neither direction: a sanity value."),
        measures=("The average rate of diffusion over the three directions, summarised over "
                  "the brain."),
        computed=(
            "From the same tensor fit as FA, with negative eigenvalues set to 0: MD is "
            "their mean, multiplied by 1000 to read in um^2/ms (10^-3 mm^2/s when the "
            "b-values are in s/mm^2). The value is the median over the brain mask."),
        reading=(
            "Neither: compare across the dataset. Tissue reads about 0.7 to 0.9 and free "
            "water (CSF) about 3 at body temperature, so a whole-brain median sits a little "
            "above the tissue range when the mask holds some CSF. A value far off for every "
            "scan of a protocol points at the b-values: a wrong unit or scale in the .bval "
            "scales MD by the same factor."),
        causes=(
            "The brain mask's share of CSF, partial volume, atrophy, and errors in the "
            "b-values; signal at the noise floor lowers the apparent diffusivity."),
        caveats=(
            "A reversed or swapped b-vector table leaves MD unchanged. b-values above the "
            "fitting limit (1500 by default) do not enter the fit."),
        references=(_BASSER94,),
    ),
    "dwi.negative_eigen": Explanation(
        title="Voxels with a negative eigenvalue",
        short=("The share of brain voxels where the fitted tensor has a negative "
               "eigenvalue, which is not physical. Lower is better."),
        measures="How often the tensor fit gives a negative diffusivity along some direction.",
        computed=("Before negative eigenvalues are set to 0, the percentage of brain voxels "
                  "whose fitted tensor has at least one below zero."),
        reading=(
            "Lower is better. A small share is usual (noise in voxels with low radial "
            "diffusivity, voxels at the edge of the mask); a clear rise against the "
            "dataset means noise, motion or artefacts larger than the diffusion contrast, "
            "or b=0 volumes that do not match the diffusion-weighted ones."),
        causes=(
            "Low SNR, motion between volumes, signal dropout, a mask that includes voxels "
            "outside the brain, or a table that does not match the volumes (for example "
            "in the wrong order)."),
        caveats=("FA computed after the clipping looks normal in these voxels; this measure "
                 "is how they are seen."),
        mriqc=(
            "Shown beside MRIQC's fa_degenerate, which targets the same failure (an "
            "unphysical fit) with a different definition; the two values are not "
            "interchangeable."),
        references=(_BASSER94,),
    ),
    "dwi.ndc": Explanation(
        title="Neighbouring direction correlation",
        short=("The mean correlation of each diffusion-weighted volume with its nearest "
               "neighbour in q-space (Yeh 2019). Higher is better."),
        measures=(
            "Neighbouring directions should give similar images; a volume corrupted by "
            "motion, dropout or artefacts correlates less with its neighbour."),
        computed=(
            "Each diffusion-weighted volume's neighbour is the volume nearest it in q-space "
            "(the square root of the b-value times the direction, either sign). The "
            "correlation is Pearson's over the brain voxels (every k-th voxel in a large "
            "brain); the value is the mean over the diffusion-weighted volumes."),
        reading=(
            "Higher is better, against the same protocol: it falls at higher b-values "
            "(lower SNR) and with fewer directions (neighbours further apart). The "
            "'Nearest-direction correlation' plot shows which volumes fall below the rest."),
        causes=("Motion, signal dropout, spikes and low SNR lower it, and so does "
                "misregistration between volumes."),
        caveats=(
            "A mean over all volumes, so one bad volume barely moves it: look at the plot. "
            "Not comparable between protocols with different b-values or numbers of "
            "directions."),
        mriqc="Same idea as MRIQC's ndc; values differ with the brain mask and the sampling.",
        references=(_YEH,),
    ),
}

_CC_SNR_CAVEAT = (
    "Needs a tensor fit and at least 5 callosal voxels (FA above 0.4, principal direction "
    "within about 32 degrees of left-right, within 10 mm of the brain's midline). The "
    "corpus callosum is deep in the head, where a multi-channel coil is least sensitive, "
    "so its SNR is not the brain's best.")

_DWI_NOISE: dict[str, Explanation] = {
    "dwi.sigma_b0": Explanation(
        title="Noise from the b=0 repeats",
        short=("The noise SD from the variation across repeated b=0 volumes, in scanner "
               "units. Neither direction on its own; motion adds to it."),
        measures="How much the same voxel varies between b=0 volumes that should be identical.",
        computed=(
            "In the brain mask eroded by two voxels, each voxel's standard deviation (with "
            "n - 1) across the b=0 volumes; the value is the median over voxels. Needs "
            "three b=0 volumes or more."),
        reading=(
            "Neither on its own: it is in the scanner's units, so read it through the SNR "
            "measures, or against scans of the same protocol, where lower is better. "
            "Clearly above the MP-PCA estimate, it says the b=0 volumes differ by more "
            "than noise: motion, drift or pulsation."),
        causes=("Thermal noise, plus motion between the b=0 volumes, signal drift, and "
                "cardiac pulsation (near the ventricles and the brainstem especially)."),
        caveats=(
            "An upper bound on the noise when the head moves. Few b=0 volumes give a noisy "
            "estimate. It is used for the SNRs only when MP-PCA could not be computed."),
    ),
    "dwi.sigma_pca": Explanation(
        title="Noise by MP-PCA",
        short=("The noise SD by Marchenko-Pastur PCA (Veraart 2016) on sampled patches of "
               "the largest shell. Neither direction on its own: it sets the SNRs."),
        measures=(
            "The thermal noise level, separated from the signal by the statistics of "
            "random matrices: the part of a patch's variation across volumes that behaves "
            "like pure noise."),
        computed=(
            "Up to 300 patches (5 x 5 x 5 voxels, or 7 x 7 x 7 when the shell has 100 "
            "volumes or more) are sampled in the eroded brain mask, keeping those at least "
            "80 % inside it. In each, the largest eigenvalues of the voxels-by-volumes "
            "matrix are set aside as signal, one by one, until the rest spread as the "
            "Marchenko-Pastur law allows for pure noise; the noise variance is the mean of "
            "the rest. The value is the median over patches; it needs six volumes or more "
            "in the shell."),
        reading=(
            "Neither on its own: it is in scanner units. It is the noise the callosal SNRs "
            "and the noise-floor measure use; a value below the b=0 estimate is expected, "
            "because motion between b=0 volumes does not enter it."),
        causes=("Thermal noise: small voxels, high acceleration, few averages, low coil "
                "sensitivity."),
        caveats=(
            "It assumes noise that is independent between voxels and the same within a "
            "patch: parallel imaging, partial Fourier, interpolation and filtering at "
            "reconstruction correlate the noise and bias the estimate. At low SNR, "
            "magnitude noise is not the Gaussian noise the method assumes."),
        mriqc=("Shown beside MRIQC's sigma_pca, a noise estimate of the same family; BIDS "
               "Manager samples 300 patches of one shell, so do not expect identical values."),
        references=(_VERAART_NI, _VERAART_MRM),
    ),
    "dwi.snr_cc_b0": Explanation(
        title="SNR in the corpus callosum (b=0)",
        short="The mean b=0 signal in the corpus callosum over the noise SD. Higher is better.",
        measures=("How far the b=0 signal of one well-defined white-matter tract stands above "
                  "the noise."),
        computed=(
            "The mean over the callosal voxels and over every b=0 volume, divided by the "
            "MP-PCA noise SD, or by the b=0 estimate when MP-PCA could not be computed."),
        reading=(
            "Higher is better, against the same protocol. It is the reference the "
            "diffusion-weighted SNRs fall from."),
        causes=("Small voxels, long echo times, high acceleration, few averages, and low "
                "coil sensitivity in the centre of the head."),
        caveats=_CC_SNR_CAVEAT,
        mriqc=(
            "MRIQC reports a callosal SNR at b=0 too (snr_cc_shell0). BIDS Manager's notes "
            "record a shell-labelling defect in MRIQC's per-shell callosal SNRs, so "
            "compare with care."),
    ),
    "dwi.snr_cc_b*_along": Explanation(
        title="SNR in the corpus callosum, along the fibres",
        short=("The corpus callosum's signal in the shell's volume with the most left-right "
               "gradient, over the noise SD. Higher is better; this is the worst case."),
        measures=(
            "How far the most attenuated signal of the shell stands above the noise: "
            "diffusion is fastest along the callosal fibres, so a gradient along them "
            "attenuates the signal most."),
        computed=(
            "One volume: the shell's volume whose gradient, in world terms, is closest to "
            "left-right. Its mean over the callosal voxels, over the noise SD (MP-PCA, "
            "else the b=0 estimate)."),
        reading=(
            "Higher is better, against the same protocol. Near or below 2 (the threshold "
            "the noise-floor measure uses), the signal is at the noise floor of magnitude "
            "data, and anything that depends on it (axial diffusivity, high-b models) is "
            "biased."),
        causes="Higher b-values, smaller voxels, longer echo times, acceleration, few averages.",
        caveats=(
            "From one volume, so it is noisier than the across value. If the table's axes "
            "are swapped, the volume picked is not the one along the fibres. "
            + _CC_SNR_CAVEAT),
        references=(_GUDBJARTSSON,),
    ),
    "dwi.snr_cc_b*_across": Explanation(
        title="SNR in the corpus callosum, across the fibres",
        short=("The corpus callosum's signal in the shell's two volumes with the least "
               "left-right gradient, over the noise SD. Higher is better; the least "
               "attenuated."),
        measures=(
            "How far the least attenuated signal of the shell stands above the noise: a "
            "gradient across the callosal fibres meets the slowest diffusion."),
        computed=(
            "The mean over the callosal voxels and over the shell's two volumes whose "
            "gradient, in world terms, has the smallest left-right component, over the "
            "noise SD (MP-PCA, else the b=0 estimate)."),
        reading=(
            "Higher is better, against the same protocol. Read it beside the along value: "
            "the gap between them reflects the callosum's anisotropy, and a shell where "
            "both are low is limited by noise."),
        causes="Higher b-values, smaller voxels, longer echo times, acceleration, few averages.",
        caveats=_CC_SNR_CAVEAT,
    ),
    "dwi.noise_floor": Explanation(
        title="Signal at the noise floor",
        short=("The share of tissue signal at the highest shell below twice the noise SD, "
               "where magnitude data are biased upwards (the Rician floor). Lower is better."),
        measures=(
            "How much of the highest-b signal is too small to measure reliably. A "
            "magnitude image cannot go below the noise: where the true signal is "
            "comparable with the noise, what is stored is biased upwards."),
        computed=(
            "Over brain voxels with an MD below 1.5 um^2/ms (tissue, not CSF, whose signal "
            "is meant to vanish at high b) and over every volume of the highest shell: the "
            "percentage of values below 2 times the noise SD (MP-PCA, else the b=0 "
            "estimate)."),
        reading=(
            "Lower is better, against the same protocol. Some share is expected at a high "
            "b-value for directions along the fibres; a large share means models fitted "
            "to that shell (kurtosis, multi-compartment) are biased unless they account "
            "for the noise floor."),
        causes="High b-values, small voxels, long echo times, low coil sensitivity, few averages.",
        caveats=(
            "Twice the noise SD is a reading aid, not a sharp limit: the bias grows "
            "smoothly as the signal falls toward the noise. Needs a tensor fit and a noise "
            "estimate."),
        references=(_GUDBJARTSSON,),
    ),
}

_DWI_ARTEFACTS: dict[str, Explanation] = {
    "dwi.dropout_slices": Explanation(
        title="Slices with signal dropout",
        short=("The number of slices, across all volumes, whose signal falls far below "
               "what the tensor predicts. Lower is better; 0 is usual."),
        measures="Slices that lost signal in one volume and not in the shell's others.",
        computed=(
            "Per slice and volume, the mean over the slice's brain voxels of (log observed "
            "- log predicted). Within each shell (and among the b=0 volumes) each slice's "
            "values become a robust z across the shell's volumes. A dropout has a z below "
            "-5 by default and is at least 10 % below the slice's typical value by default "
            "(both Settings > Quality control > Diffusion); the value counts (slice, "
            "volume) pairs."),
        reading=(
            "Lower is better. Each dropout is a slice of one volume to replace or exclude; "
            "when more than 5 % of the volumes hold one, the finding is an error. The blue "
            "cells of the 'Slice signal' plot are the dropouts; in a multiband "
            "(simultaneous multi-slice) acquisition, slices acquired together drop out "
            "together."),
        causes=(
            "Bulk motion during the diffusion encoding (cardiac pulsation in the brainstem "
            "and cerebellum, or the head moving), and table vibration with strong "
            "gradients."),
        caveats=(
            "A dropout over part of a slice shows only if it lowers the slice's mean by "
            "the threshold. Slices with little brain (top and bottom) are not judged. The "
            "typical value for the 10 % test is the slice's median over every volume of "
            "every shell."),
        mriqc="Not in MRIQC.",
        references=(_ANDERSSON_OLR,),
    ),
    "dwi.interleave_volumes": Explanation(
        title="Volumes with an interleave artefact",
        short=("The number of volumes whose odd slices disagree with their even slices far "
               "more than in the rest of the shell. Lower is better; 0 is usual."),
        measures="Motion between the two passes of an interleaved slice acquisition.",
        computed=(
            "Per volume, the median residual against the tensor's prediction over odd "
            "slices minus that over even slices, as a robust z within the shell. A volume "
            "counts when |z| is above 5 by default (Settings > Quality control > "
            "Diffusion) and the difference is above 0.05 in log units (about 5 %)."),
        reading=(
            "Lower is better. Such a volume shows stripes alternating slice by slice (a "
            "venetian-blind pattern) in a sagittal or coronal view."),
        causes=("Head motion between the passes of an interleaved acquisition, which records "
                "odd and even slices at different times."),
        caveats=(
            "Meaningful for interleaved acquisitions; with multiband or sequential "
            "ordering, odd and even slices are not two passes, and a flagged volume points "
            "at another slice-wise effect."),
        mriqc="Not in MRIQC.",
    ),
    "dwi.spikes_ppm": Explanation(
        title="Spiking voxels",
        short=("Voxels far above the tensor's prediction, per million brain voxels per "
               "volume, averaged over the volumes. Lower is better."),
        measures="Single voxels much brighter than they should be in one volume.",
        computed=(
            "Each voxel's log residual (observed against predicted) in each volume of a "
            "shell is compared with its own median residual across that shell. A spike is "
            "more than 8 robust spreads above by default (Settings > Quality control > "
            "Diffusion), the spread floored at 0.05, and more than 0.5 in log units (about "
            "65 % above). Counts per volume are scaled per million brain voxels and "
            "averaged."),
        reading=("Lower is better; 0 is usual. The 'Spiking voxels' plot shows which volumes "
                 "carry them."),
        causes=(
            "RF spikes (a spike in k-space shows as a striped pattern over the slice, "
            "parts of which cross the threshold), and motion that moves bright tissue into "
            "voxels that are usually dark at high b, such as CSF."),
        caveats=(
            "An average over all volumes, so a single bad volume can look small: check the "
            "plot. Comparing each voxel with itself means the tensor's own misfit (CSF, "
            "crossing fibres) is not counted."),
        mriqc=("Not the same as MRIQC's spike measure: the module's notes record a defect in "
               "MRIQC's spike mask, so the two are not comparable."),
    ),
    "dwi.efc_b*": Explanation(
        title="Entropy focus criterion (per shell)",
        short=("The entropy focus criterion of the shell's mean image (b=0: the median of "
               "the b=0 volumes). Lower is better."),
        measures=("How evenly the energy of the shell's mean image is spread: ghosting and "
                  "blurring spread it into the background."),
        computed=(
            "As for anatomical images, over every voxel except the zero fill of the "
            "shell's mean image: E = sum (x/B) ln(x/B) with B = sqrt(sum x^2), divided by "
            "its value for a uniform image."),
        reading=(
            "Lower is better, against the same shell of scans of the same protocol. Higher "
            "b-values give darker images and different values, so compare shell with "
            "shell."),
        causes=("Nyquist ghosts of the echo-planar readout, motion blur across the averaged "
                "volumes, and background noise."),
        caveats=(
            "Not computed when the sidecar records defacing or the field of view holds "
            "fewer than 500 air voxels; the diffusion check does not detect a blank face "
            "or a skull-stripped image by itself. Averaging more volumes lowers the mean "
            "image's background noise and changes the value."),
        mriqc="Same formula as MRIQC's efc.",
        references=(_ATKINSON,),
    ),
    "dwi.fber_b*": Explanation(
        title="Foreground to background energy (per shell)",
        short=("The head's typical energy over the air's in the shell's mean image. Higher is "
               "better."),
        measures="How much brighter the head is than its background in the shell's mean image.",
        computed=(
            "The median of the squared intensities inside the head mask (made on the b=0 "
            "image) over the median outside it by two voxels, zero fill excluded, on the "
            "shell's mean image (b=0: the median of the b=0 volumes)."),
        reading=(
            "Higher is better, shell by shell, against the same protocol: diffusion "
            "weighting darkens the head, so FBER falls with b."),
        causes="Nyquist ghosts and any other signal in the background lower it.",
        caveats=(
            "The same conditions as EFC. The air is all the background outside the head, "
            "face and neck included, because the diffusion check registers no template."),
        mriqc="Same formula as MRIQC's fber.",
        references=(_SHEHZAD,),
    ),
}


# ---------------------------------------------------------------------------
# The QC plots under a BOLD time course
# ---------------------------------------------------------------------------

_BOLD_MOTION_SOURCE = (
    "From fMRIPrep's confounds table when the run has one of the right length, else "
    "estimated here: each volume registered rigidly to the first volume after any "
    "non-steady-state ones, on a grid of about 3.5 mm by default (Settings > Quality "
    "control > Functional time series (BOLD)), using the reference's 15,000 strongest "
    "edges. Settings > Quality control > Methods chooses whether the confounds are read.")

_BOLD_PLOTS: dict[str, Explanation] = {
    "plot.bold.fd": Explanation(
        title="Framewise displacement",
        short=("How far the head moved from one volume to the next, in mm (Power 2012). "
               "Lower is better; volumes above the dashed line are marked."),
        measures=(
            "Frame-to-frame head movement as one number per volume: the absolute changes "
            "of the three translations plus those of the three rotations, the rotations as "
            "arcs on a 50 mm sphere by default (Settings > Quality control > Functional "
            "time series (BOLD))."),
        computed=(
            _BOLD_MOTION_SOURCE + " With the confounds, their framewise_displacement "
            "column is shown, recomputed from their parameters when the radius is set to "
            "other than 50 mm. The first volume has no FD."),
        reading=(
            "Lower is better. The dashed line is 0.5 mm by default (Settings > Quality "
            "control > Functional time series (BOLD)), Power 2012's and fMRIPrep's usual "
            "threshold; stricter analyses use lower ones. Isolated peaks are sudden "
            "movements, a raised baseline is restlessness throughout. The note says "
            "whether the numbers come from the confounds or from the estimate."),
        causes=(
            "Head motion, swallowing, coughing and talking; with short repetition times "
            "also the apparent motion breathing causes, which shifts the image along the "
            "phase-encoding direction."),
        caveats=(
            "The estimate is a screening registration: against fMRIPrep's it agreed "
            "closely for a participant who moved (r 0.94 to 0.98) but read about 1.4 times "
            "higher for a still one, whose motion is near the registration's noise. FD per "
            "volume depends on the repetition time: the same movement spread over more, "
            "shorter volumes gives smaller values, so a threshold set for one TR does not "
            "carry to another."),
        mriqc=("MRIQC summarises the same FD as fd_mean and counts volumes above its own "
               "threshold (fd_num, fd_perc)."),
        references=(_POWER12,),
    ),
    "plot.bold.translation": Explanation(
        title="Translation (x, y, z)",
        short=("Where the head is along three axes, in mm, relative to the reference volume. "
               "Flat near zero is best."),
        measures="The head's position over the run along three axes.",
        computed=(
            "The three translations of the same motion as FD (fMRIPrep's trans_x, trans_y, "
            "trans_z, or the estimate). For the estimate the reference is the first steady "
            "volume and the axes are the image's voxel axes, which are left-right, "
            "back-front and down-up for the usual axial acquisition."),
        reading=(
            "Neither direction: read the shape. A slow drift over the run is common (the "
            "head settling); a step is a sudden movement; a total range larger than a "
            "voxel means the same voxel holds different tissue at different times."),
        causes="Head motion and slow settling; with short repetition times, breathing.",
        caveats=("Relative to a reference: the values depend on which volume that is, not "
                 "only on how the head moved."),
    ),
    "plot.bold.rotation": Explanation(
        title="Rotation (pitch, roll, yaw)",
        short=("How the head is turned, in degrees, relative to the reference volume. Flat "
               "near zero is best."),
        measures=(
            "Pitch about the left-right axis (nodding), roll about the back-front axis "
            "(tilting to a shoulder), yaw about the vertical axis (shaking the head)."),
        computed=(
            "The three rotations of the same motion as FD, in degrees (fMRIPrep's rot_x, "
            "rot_y, rot_z, or the components of the estimate's rotation vector). For the "
            "estimate the axes are the image's voxel axes, which match those names for "
            "the usual axial acquisition."),
        reading=(
            "Neither direction: read the shape. One degree moves a point 50 mm from the "
            "centre of rotation by about 0.87 mm, so small angles matter at the edge of "
            "the brain. Nodding (pitch) is the commonest."),
        causes="Head motion, most often nodding; swallowing and breathing.",
        caveats="Relative to a reference volume, as the translations are.",
    ),
    "plot.bold.dvars": Explanation(
        title="DVARS",
        short=("How much the whole image changes from one volume to the next, in percent of "
               "the mean signal. Lower is better; volumes above the dashed fence are marked."),
        measures="The size of the change between consecutive volumes, over the head.",
        computed=(
            "Over the head mask of a reference volume (voxels brighter than a fifth of its "
            "98th percentile): the root mean square of each volume minus the previous one, "
            "over the run's median global signal, times 100. The dashed fence is the 75th "
            "percentile plus 1.5 interquartile ranges of the run's own DVARS by default "
            "(Settings > Quality control > Functional time series (BOLD)), FSL's rule. "
            "Non-steady-state volumes at the start, left out by default (same section), "
            "do not enter the fence or the marks; volume 0 has no value."),
        reading=(
            "Lower is better. Peaks that coincide with FD peaks are motion; peaks without "
            "FD point at spikes, slice dropouts or sudden physiological changes. Read it "
            "with FD and the carpet plot."),
        causes=("Motion, RF spikes, slice dropouts, deep breaths, and non-steady-state "
                "volumes at the start."),
        caveats=(
            "The fence comes from the run itself, so a run full of motion has a high "
            "fence, and a clean run still marks its largest values. It is not standardised, "
            "so its scale depends on the run's noise and does not compare between "
            "protocols."),
        mriqc=("Not on the scale of MRIQC's dvars_std, dvars_nstd or dvars_vstd, which are "
               "standardised."),
        references=(_POWER12, _AFYOUNI),
    ),
    "plot.bold.outliers": Explanation(
        title="Outlier voxels",
        short=("The percentage of the head's voxels that are outliers in each volume "
               "(AFNI's 3dToutcount rule). Lower is better."),
        measures="How many voxels are far from their own usual values at the same moment.",
        computed=(
            "8000 voxels by default (Settings > Quality control > Functional time series "
            "(BOLD)) are sampled from the head mask and each voxel's series has a quadratic "
            "trend removed. A value is an outlier when it lies further from its voxel's "
            "median than qginv(0.001/N) x sqrt(pi/2) x MAD, N the number of volumes. The "
            "dashed line is 5 % by default (same section), afni_proc.py's censoring "
            "default."),
        reading=(
            "Lower is better. A volume above the line has many voxels far from their usual "
            "values at once: motion, a spike or a dropout."),
        causes="Motion, RF spikes, slice dropouts, and non-steady-state volumes.",
        caveats=(
            "From a sample of voxels, so small, local events can be missed. Voxels at the "
            "head's edge become outliers with small movements, so motion dominates the plot."),
        mriqc=("The same rule as MRIQC's aor (AFNI's outlier ratio), shown per volume where "
               "MRIQC reports the mean."),
        references=(_COX,),
    ),
    "plot.bold.spikes": Explanation(
        title="Slice spikes",
        short=("The largest robust z of any slice against its own course, per volume. Lower "
               "is better; volumes above the dashed line are marked."),
        measures="Single slices that depart from their own course in one volume.",
        computed=(
            "Each slice's mean over the head, along the slice axis (the sidecar's "
            "SliceEncodingDirection, else the third axis). What the whole volume did (an "
            "offset and a scaling of the run's typical slice profile) is removed, then a "
            "quadratic trend over time; each slice becomes a robust z across volumes, and "
            "the plot shows the largest |z| per volume. The dashed line is 6 robust SD by "
            "default (Settings > Quality control > Functional time series (BOLD))."),
        reading=(
            "Lower is better. A marked volume has one slice that departed from its own "
            "course; the summary names the slice, so check that volume and slice in the "
            "image."),
        causes=("RF spikes, a slice spoiled by motion during the volume (interleaved or "
                "multiband acquisitions), and artefacts confined to single slices."),
        caveats=("Whole-volume changes are removed on purpose, so global signal changes and "
                 "slow drift are not spikes."),
    ),
    "plot.bold.global": Explanation(
        title="Global signal",
        short="The mean signal over the head, per volume. Neither direction: read the shape.",
        measures="How the overall signal level changes through the run.",
        computed=(
            "The mean over the head mask (voxels brighter than a fifth of a reference "
            "volume's 98th percentile) in each volume, in the scanner's units."),
        reading=(
            "Look for steps (a scanner adjustment or a large movement), a bright start "
            "(non-steady-state volumes the scanner did not discard), slow drift, and slow "
            "oscillations. Read it with FD and DVARS: a change seen in all three is motion."),
        causes=("Scanner drift and heating, non-steady-state volumes, motion, and "
                "physiology: breathing, heart rate and arousal all move it."),
        caveats="Not a noise measure: much of its slow variation is physiological and real.",
    ),
    "plot.bold.carpet": Explanation(
        title="Carpet plot",
        short=("Every sampled voxel's signal over time, one row each, from the edge of the "
               "head inwards (Power 2017). Vertical bands are events across the head."),
        measures=("The whole run at a glance: what happened to every part of the head at every "
                  "volume."),
        computed=(
            "8000 sampled head voxels by default (Settings > Quality control > Functional "
            "time series (BOLD)), each with a quadratic trend removed and scaled to its own "
            "SD (clipped at 3), ordered by distance from the head's edge (the edge at the "
            "top) and averaged into 240 rows. Colours span z from -2 to 2; non-steady-state "
            "volumes, left out by default, are blank."),
        reading=(
            "A clean run looks like even texture. Vertical bands across many rows are "
            "events that touched the whole head at once: motion, a deep breath, a spike. "
            "Bands strongest at the top (the edge) are typical of motion."),
        causes="Motion, breathing and arousal changes, RF spikes, and slice dropouts.",
        caveats=(
            "Rows are ordered by depth, not by tissue as in Power's original, so grey "
            "matter, white matter and CSF mix in each row. Each voxel is scaled to its own "
            "SD, so differences in amplitude between voxels do not show."),
        mriqc="MRIQC's functional report draws a carpet plot too, ordered by tissue.",
        references=(_POWER17,),
    ),
}


# ---------------------------------------------------------------------------
# The QC plots under a diffusion time course
# ---------------------------------------------------------------------------

_DWI_PLOTS: dict[str, Explanation] = {
    "plot.dwi.directions": Explanation(
        title="Directions on the sphere",
        short=("Each diffusion volume's gradient direction, one colour per shell, seen along "
               "one axis. An even spread covers the sphere; gaps and clusters do not."),
        measures=(
            "Where the gradient directions of each shell point. A direction and its "
            "opposite measure the same thing, so every direction is folded onto the half of "
            "the sphere facing the viewer."),
        computed=(
            "Each b-vector is normalised and reversed when it points away from the viewing "
            "axis, then drawn with Lambert's equal-area projection, so an even spread looks "
            "even: the centre is the viewing axis, the outer circle the plane across it, "
            "and the faint rings lie 30 and 60 degrees from the axis. The b=0 volumes have "
            "no direction and are not drawn."),
        reading=(
            "Look for empty regions and for points on top of one another, and view along "
            "each axis: a scheme can look even from one side and leave a gap seen from "
            "another. The largest gap of each shell, beside its name, puts a number on it."),
        causes=(
            "A poorly designed scheme, a scheme meant for a full sphere acquired as half of "
            "one, and volumes dropped at conversion or by a scan stopped early."),
        caveats=(
            "A property of the table, not of the image: an image whose table is wrong or "
            "flipped looks the same here. The axes are the image's, as the .bvec stores "
            "them."),
        references=(_JONES99, _JEURISSEN),
    ),
    "plot.dwi.bvalues": Explanation(
        title="b-value of each volume",
        short=("Every volume's b-value in the order acquired, one colour per shell and the "
               "b=0 volumes in grey. Shows how the b=0 volumes are spread through the scan."),
        measures="The order in which the shells and the b=0 volumes were acquired.",
        computed=(
            "Each volume's b-value from the .bval, at its position in the series. Values up "
            "to the b=0 limit count as b=0 and the others are grouped into shells rounded to "
            "the nearest 50, as the quality check groups them (Settings > Quality control > "
            "Diffusion)."),
        reading=(
            "b=0 volumes spread through the series let drift and motion be followed over "
            "the whole scan; all of them at the start cannot. Interleaved shells share "
            "heating and motion evenly. A point off its shell's line is a b-value unlike "
            "its neighbours'."),
        causes="The acquisition protocol; series reordered or joined after the scan.",
        caveats="It shows the table, not whether the images were acquired in that order.",
        references=(_VOS,),
    ),
    "plot.dwi.displacement": Explanation(
        title="Displacement",
        short=("How far each volume sits from the first b=0, in mm, with volumes far out of "
               "place marked. Lower is better."),
        measures="The head's position over the series, as one distance per volume.",
        computed=(
            "The three absolute translations plus the three absolute rotations as arcs on "
            "a 50 mm sphere by default, each relative to the first b=0, with eddy-current "
            "shifts removed when the gradients explain them. The dashed line is 2 mm by "
            "default (Settings > Quality control > Diffusion); a marked volume is beyond it "
            "and a robust outlier of its shell."),
        reading=(
            "Lower is better. The b=0 volumes are the head's reliable track; "
            "diffusion-weighted volumes scatter around it by the registration's noise, "
            "more at higher b. Follow the b=0 points for slow motion and look for single "
            "volumes far above their neighbours."),
        causes="Head motion; registration noise on low-signal volumes.",
        caveats=(
            "A displacement from the first b=0, not a frame-to-frame change: a head that "
            "moved once early and stayed there reads high for the rest of the scan."),
        references=(_POWER12,),
    ),
    "plot.dwi.translation": Explanation(
        title="Translation (x, y, z)",
        short=("Where each volume sits against the first b=0, in mm, along the image's three "
               "axes. Flat near zero is best."),
        measures="The head's position over the series along three axes.",
        computed=(
            "The three translations of each volume relative to the first b=0, along the "
            "image's voxel axes, with the eddy-current part removed when the gradients "
            "explain at least 30 % of a shell's translations by default (Settings > "
            "Quality control > Diffusion)."),
        reading=(
            "Neither direction: read the shape. A slow drift is the head settling, a step "
            "a movement; scatter that follows the diffusion-weighted volumes and not the "
            "b=0 ones is registration noise or eddy current left in."),
        causes="Head motion, eddy currents, and registration noise.",
        caveats="A screening registration, not the correction an analysis uses.",
    ),
    "plot.dwi.rotation": Explanation(
        title="Rotation (pitch, roll, yaw)",
        short=("How each volume is turned against the first b=0, in degrees. Flat near zero "
               "is best."),
        measures=("How the head is turned over the series, and so how the gradients turned "
                  "with it."),
        computed=(
            "The components of each volume's rotation vector relative to the first b=0, "
            "about the image's first, second and third voxel axes (pitch, roll and yaw for "
            "the usual axial acquisition), in degrees."),
        reading=(
            "Neither direction: read the shape. Rotations above a few degrees mean the "
            "b-vectors should be rotated with the motion correction."),
        causes="Head rotation, most often nodding.",
        caveats="A screening registration: small values are within its noise.",
        references=(_LEEMANS,),
    ),
    "plot.dwi.slices": Explanation(
        title="Slice signal",
        short=("Every slice of every volume against what the tensor predicts, as a robust z "
               "within its shell. Blue is signal lost (a dropout), red is signal added."),
        measures=("Where in the series, slice by slice, the signal departs from the tensor's "
                  "prediction."),
        computed=(
            "Rows are slices (the top slice at the top), columns volumes. Each cell is the "
            "slice's mean log residual (observed against predicted) as a robust z among "
            "the volumes of its shell, clipped at 10 either way and coloured from -8 to 8; "
            "slices with too little brain are blank. Dropouts found by the check are "
            "flagged, and a click goes to that volume and slice."),
        reading=(
            "A single blue cell is a dropout in one slice of one volume. A blue column is "
            "a whole volume that lost signal, a red one a volume that gained it. A pattern "
            "repeating every few rows in one column points at multiband slices acquired "
            "together."),
        causes=("Motion during the diffusion encoding, cardiac pulsation, table vibration, "
                "and interleave artefacts."),
        caveats=("The z is relative to the shell: a slice that is bad in every volume of a "
                 "shell does not stand out."),
        references=(_ANDERSSON_OLR,),
    ),
    "plot.dwi.b0_signal": Explanation(
        title="b=0 signal",
        short=("The brain's median signal in each b=0 volume against the first, in percent. "
               "A slope is drift."),
        measures="How the scanner's signal changes over the series, as the b=0 volumes show it.",
        computed=(
            "The median over the brain mask of each b=0 volume, over the first b=0's, "
            "times 100, drawn at each b=0's position in the series. The drift measure is a "
            "fit through these points."),
        reading=(
            "Flat at 100 is best. A steady slope is drift; a single point off the line is "
            "motion or an artefact in that volume."),
        causes="Gradient and shim heating; motion.",
        caveats=("Only as many points as b=0 volumes; with all of them at the start, it says "
                 "nothing about the end of the scan."),
        references=(_VOS,),
    ),
    "plot.dwi.spikes": Explanation(
        title="Spiking voxels",
        short=("Voxels far above the tensor's prediction, per million brain voxels, per "
               "volume. Lower is better."),
        measures="Which volumes hold single voxels much brighter than they should be.",
        computed=(
            "Each voxel's residual against the tensor's prediction is compared with its "
            "own median across the shell; a spike is more than 8 robust spreads above by "
            "default (Settings > Quality control > Diffusion) and more than about 65 % "
            "above. The count per volume is scaled per million brain voxels."),
        reading="Lower is better; peaks are volumes to inspect.",
        causes="RF spikes; motion that moves bright tissue into usually dark voxels.",
        caveats="Volumes of a shell with fewer than four volumes are not judged.",
    ),
    "plot.dwi.interleave": Explanation(
        title="Odd against even slices",
        short=("Per volume, odd slices against even slices relative to the tensor's "
               "prediction, as a robust z within the shell. Near zero is best."),
        measures="Motion between the two passes of an interleaved slice acquisition.",
        computed=(
            "Per volume, the median residual over odd slices minus that over even slices, "
            "as a robust z within the shell. The dashed line is 5 by default (Settings > "
            "Quality control > Diffusion); a volume is marked when |z| is above it and "
            "the difference is above about 5 % in signal."),
        reading=(
            "Near zero is best; the mark uses the size of z, so a volume far below zero is "
            "marked as well as one far above. A marked volume shows a venetian-blind "
            "pattern in a sagittal or coronal view."),
        causes="Head motion between the passes of an interleaved acquisition.",
        caveats="Meaningful for interleaved acquisitions; with other slice orders it points "
                "at another slice-wise effect.",
    ),
    "plot.dwi.ndc": Explanation(
        title="Nearest-direction correlation",
        short=("Each diffusion-weighted volume's correlation with the volume nearest it in "
               "q-space. Higher is better."),
        measures="Whether each volume looks like its nearest neighbour in direction and b-value.",
        computed=(
            "Pearson's correlation over sampled brain voxels between each "
            "diffusion-weighted volume and its nearest neighbour in q-space (the square "
            "root of the b-value times the direction, either sign); b=0 volumes have none. "
            "Volumes with a robust z below -5 among all diffusion-weighted volumes (not "
            "per shell) are marked."),
        reading=(
            "Higher is better. Points far below the rest are volumes unlike their "
            "neighbours: check them in the 'Slice signal' plot and the image. With several "
            "shells, each sits at its own level, lower at higher b."),
        causes="Motion, signal dropout, spikes, low SNR, misregistration between volumes.",
        caveats=(
            "With several shells the marks are judged over all diffusion-weighted volumes "
            "together, so a high shell that is a minority of the volumes can be marked as "
            "a whole for its naturally lower correlation."),
        references=(_YEH,),
    ),
    "plot.dwi.eddy": Explanation(
        title="Eddy-current shift",
        short=("The shift of each diffusion-weighted volume its gradient direction explains, "
               "in mm, along the image's three axes."),
        measures=("The part of each volume's displacement that eddy currents, not the head, "
                  "caused."),
        computed=(
            "In each shell of at least 10 volumes, the translations are regressed on the "
            "gradient's components; the explained part is shown when it accounts for at "
            "least 30 % of the shell's translations by default (Settings > Quality control "
            "> Diffusion). The plot appears only when some shell passes."),
        reading=(
            "Lower is better. The shift is usually largest along the phase-encoding axis "
            "and follows the gradient direction from volume to volume; a voxel or more "
            "calls for distortion correction, which is routine."),
        causes="Strong, fast-switching diffusion gradients and high b-values.",
        caveats="Only the shift; the stretch and shear eddy currents cause are not shown.",
        references=(_ANDERSSON_EDDY,),
    ),
}


# ---------------------------------------------------------------------------
# BOLD quality maps
# ---------------------------------------------------------------------------

_BOLD_MAPS: dict[str, Explanation] = {
    "map.bold.mean": Explanation(
        title="Mean",
        short=("The mean of each voxel through the run: a picture of the anatomy and of "
               "where signal is lost."),
        measures="Where the signal is, and where it is missing.",
        computed=(
            "The mean over the run's volumes, without the non-steady-state volumes at the "
            "start when Settings > Quality control > Functional time series (BOLD) leaves "
            "them out (the default). Shown inside the head: voxels above a fifth of the "
            "map's 98th percentile."),
        reading=(
            "Look for dark regions inside the head: susceptibility dropout near the "
            "sinuses and ear canals (orbitofrontal cortex, temporal poles), and brightness "
            "that falls from the edge to the centre (receive-coil sensitivity). The "
            "temporal SNR is low where the mean is dark, for that reason rather than noise."),
        causes=("Susceptibility differences at air-tissue boundaries, coil sensitivity, "
                "slice coverage, and wrap-around."),
        caveats=(
            "In scanner units: compare patterns, not values. Voxels below a fifth of the "
            "98th percentile are hidden, so faint ghosts outside the head do not show."),
    ),
    "map.bold.sd": Explanation(
        title="Standard deviation",
        short=("The standard deviation of each voxel through the run, around a slow trend. "
               "Bright means the signal moves."),
        measures="Where the signal varies over time.",
        computed=(
            "Around a quadratic trend by default, after the non-steady-state volumes "
            "(Settings > Quality control > Functional time series (BOLD)), in the "
            "scanner's units; shown where the mean map is shown."),
        reading=(
            "Bright in vessels, ventricles and the brainstem is expected (pulsation). A "
            "bright rim at the brain's edge means motion; bright copies of the head along "
            "the phase-encoding direction mean ghosting."),
        causes="Motion, cardiac and respiratory pulsation, ghosting, and thermal noise.",
        caveats=(
            "In scanner units: compare patterns within a run, not values between scanners. "
            "It is hidden wherever the mean is below a fifth of its 98th percentile, so a "
            "ghost shows only where it overlaps the displayed head."),
    ),
    "map.bold.tsnr": Explanation(
        title="Temporal SNR",
        short=("Each voxel's mean over its standard deviation through the run, after slow "
               "drift is removed. Higher is better."),
        measures=("How steady each voxel's signal is: how much of it is signal rather than "
                  "fluctuation."),
        computed=(
            "The mean map over the standard deviation map (both after the non-steady-state "
            "volumes, the SD around a quadratic trend by default). The summary gives the "
            "median over the head and the share of the head below 20 by default (both "
            "Settings > Quality control > Functional time series (BOLD))."),
        reading=(
            "Higher is steadier. Compare runs of the same protocol: it depends on field "
            "strength, voxel size, acceleration, the coil and the repetition time, so no "
            "single number is a pass mark. Grey matter usually reads lower than white "
            "matter, because physiological fluctuations are larger there. Patterns: a low "
            "ring at the brain's edge (motion), low values in ventricles and along vessels "
            "(pulsation), low where the mean is dark (dropout), alternating slices "
            "(slice-wise motion or spikes)."),
        causes=("Thermal noise (small voxels, high acceleration), physiological noise "
                "(breathing, heartbeat), motion, and signal dropout."),
        caveats=(
            "It does not grow in step with image SNR: as voxels get larger or the field "
            "stronger, physiological fluctuations take over and it levels off. It is "
            "computed on the data as stored, before motion correction, so motion lowers it."),
        mriqc=("MRIQC reports tsnr, the median temporal SNR in the brain; BIDS Manager's "
               "median is over the displayed head region, so the values differ."),
        references=(_KRUEGER, _TRIANTAFYLLOU),
    ),
}


#: Every explanation, by key: ``anat.<metric>``, ``dwi.<metric>`` (per-shell
#: metrics as ``*`` patterns), ``group.<kind>.<group>``,
#: ``plot.<bold|dwi>.<row id>`` and ``map.bold.<map>``.
ENTRIES: dict[str, Explanation] = {
    **_ANAT_GROUPS,
    **_ANAT_NOISE,
    **_ANAT_ARTEFACTS,
    **_ANAT_TISSUES,
    **_ANAT_COVERAGE,
    **_DWI_GROUPS,
    **_DWI_GRADIENTS,
    **_DWI_MOTION,
    **_DWI_DIFFUSION,
    **_DWI_NOISE,
    **_DWI_ARTEFACTS,
    **_BOLD_PLOTS,
    **_DWI_PLOTS,
    **_BOLD_MAPS,
}


def lookup(key: str) -> Optional[Explanation]:
    """Exact key first, then a pattern with ``*`` (``dwi.efc_b*``)."""
    found = ENTRIES.get(key)
    if found is not None:
        return found
    for pattern, entry in ENTRIES.items():
        if "*" in pattern and fnmatchcase(key, pattern):
            return entry
    return None


__all__ = ["ENTRIES", "Explanation", "lookup"]
