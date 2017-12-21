#include <otf2/otf2.h>
#include <stdio.h>
#include <stdlib.h>

enum /* command line arguments */
{
    ARG0,
    TRACENAME,
    NARGS
};

typedef struct AnalysisData_st
{
    OTF2_Reader *reader;
    double *comm_matrix;
    unsigned matrix_size;
} AnalysisData;

static OTF2_CallbackCode
loc(void *userData,
    OTF2_LocationRef location,
    OTF2_StringRef name,
    OTF2_LocationType locationType,
    uint64_t numberOfEvents,
    OTF2_LocationGroupRef locationGroup)
{
    AnalysisData *data = userData;
    OTF2_Reader *reader = data->reader;
    OTF2_DefReader *def_reader = OTF2_Reader_GetDefReader(reader,
                                                          location);
    if (def_reader) {
        uint64_t def_reads = 0;
        OTF2_Reader_ReadAllLocalDefinitions(reader,
                                            def_reader,
                                            &def_reads);
        OTF2_Reader_CloseDefReader(reader, def_reader);
    }
    OTF2_Reader_GetEvtReader(reader, location);
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
str(void *userData, OTF2_StringRef self, const char *string)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
clock_properties(void *userData,
                 uint64_t timerResolution,
                 uint64_t globalOffset,
                 uint64_t traceLength)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
region(void *userData,
       OTF2_RegionRef self,
       OTF2_StringRef name,
       OTF2_StringRef canonicalName,
       OTF2_StringRef description,
       OTF2_RegionRole regionRole,
       OTF2_Paradigm paradigm,
       OTF2_RegionFlag regionFlags,
       OTF2_StringRef sourceFile,
       uint32_t beginLineNumber,
       uint32_t endLineNumber)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
enter(OTF2_LocationRef locationID,
      OTF2_TimeStamp time,
      void *userData,
      OTF2_AttributeList *attributeList,
      OTF2_RegionRef region)
{
    return OTF2_CALLBACK_SUCCESS;
}


static OTF2_CallbackCode
leave(OTF2_LocationRef locationID,
      OTF2_TimeStamp time,
      void *userData,
      OTF2_AttributeList *attributeList,
      OTF2_RegionRef region)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpisend(OTF2_LocationRef locationID,
        OTF2_TimeStamp time,
        void *userData,
        OTF2_AttributeList *attributeList,
        uint32_t receiver,
        OTF2_CommRef communicator,
        uint32_t msgTag,
        uint64_t msgLength)
{
    AnalysisData *data = userData;
    data->comm_matrix[locationID * data->matrix_size + receiver] +=
        (double) msgLength;
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpiisend(OTF2_LocationRef locationID,
         OTF2_TimeStamp time,
         void *userData,
         OTF2_AttributeList *attributeList,
         uint32_t receiver,
         OTF2_CommRef communicator,
         uint32_t msgTag,
         uint64_t msgLength,
         uint64_t requestID)
{
    AnalysisData *data = userData;
    data->comm_matrix[locationID * data->matrix_size + receiver] +=
        (double) msgLength;
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpiisendcomplete(OTF2_LocationRef locationID,
                 OTF2_TimeStamp time,
                 void *userData,
                 OTF2_AttributeList *attributeList,
                 uint64_t requestID)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpirecv(OTF2_LocationRef locationID,
        OTF2_TimeStamp time,
        void *userData,
        OTF2_AttributeList *attributeList,
        uint32_t sender,
        OTF2_CommRef communicator,
        uint32_t msgTag,
        uint64_t msgLength)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpiirecv(OTF2_LocationRef locationID,
         OTF2_TimeStamp time,
         void *userData,
         OTF2_AttributeList *attributeList,
         uint32_t sender,
         OTF2_CommRef communicator,
         uint32_t msgTag,
         uint64_t msgLength,
         uint64_t requestID)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpiirecvrequest(OTF2_LocationRef locationID,
                OTF2_TimeStamp time,
                void *userData,
                OTF2_AttributeList *attributeList,
                uint64_t requestID)
{
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode
mpicollectiveend(OTF2_LocationRef locationID,
                 OTF2_TimeStamp time,
                 void *userData,
                 OTF2_AttributeList *attributeList,
                 OTF2_CollectiveOp collectiveOp,
                 OTF2_CommRef communicator,
                 uint32_t root,
                 uint64_t sizeSent,
                 uint64_t sizeReceived)
{
    return OTF2_CALLBACK_SUCCESS;
}

static void
read_global_definitions(OTF2_Reader *reader, AnalysisData *data)
{
    uint64_t definitions_read = 0;
    OTF2_GlobalDefReader *global_def_reader = NULL;
    OTF2_GlobalDefReaderCallbacks *global_def_callbacks = NULL;
    global_def_reader = OTF2_Reader_GetGlobalDefReader(reader);
    global_def_callbacks = OTF2_GlobalDefReaderCallbacks_New();
    OTF2_GlobalDefReaderCallbacks_SetLocationCallback(
                                                   global_def_callbacks,
                                                   &loc);
    OTF2_GlobalDefReaderCallbacks_SetStringCallback(global_def_callbacks,
                                                    &str);
    OTF2_GlobalDefReaderCallbacks_SetRegionCallback(global_def_callbacks,
                                                    &region);
    OTF2_GlobalDefReaderCallbacks_SetClockPropertiesCallback(
                                                global_def_callbacks,
                                                &clock_properties);
    OTF2_Reader_RegisterGlobalDefCallbacks(reader,
                                           global_def_reader,
                                           global_def_callbacks,
                                           data);
    OTF2_GlobalDefReaderCallbacks_Delete(global_def_callbacks);
    global_def_callbacks = NULL;
    OTF2_Reader_ReadAllGlobalDefinitions(reader,
                                         global_def_reader,
                                         &definitions_read);    
}

static void
read_events(OTF2_Reader *reader, AnalysisData *data)
{
    uint64_t events_read = 0;
    OTF2_GlobalEvtReader *global_evt_reader = NULL;
    OTF2_GlobalEvtReaderCallbacks *event_callbacks = NULL;
    global_evt_reader = OTF2_Reader_GetGlobalEvtReader(reader);
    event_callbacks = OTF2_GlobalEvtReaderCallbacks_New();
    OTF2_GlobalEvtReaderCallbacks_SetEnterCallback(event_callbacks,
                                                   &enter);
    OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback(event_callbacks,
                                                   &leave);
    OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback(event_callbacks,
                                                     &mpisend);
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback(event_callbacks,
                                                      &mpiisend);
    OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback(
                                                     event_callbacks,
                                                     &mpiisendcomplete);
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback(
                                                       event_callbacks,
                                                       &mpiirecvrequest);
    OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback(event_callbacks,
                                                     &mpirecv);
    OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback(event_callbacks,
                                                      &mpiirecv);
    OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback(
                                                      event_callbacks,
                                                      &mpicollectiveend);
    OTF2_Reader_RegisterGlobalEvtCallbacks(reader,
                                           global_evt_reader,
                                           event_callbacks,
                                           data);
    OTF2_GlobalEvtReaderCallbacks_Delete(event_callbacks);
    event_callbacks = NULL;
    OTF2_Reader_ReadAllGlobalEvents(reader,
                                    global_evt_reader,
                                    &events_read);
    OTF2_Reader_CloseGlobalEvtReader(reader, global_evt_reader);
    global_evt_reader = NULL;
}

int
main(int argc, char **argv)
{
    OTF2_Reader *reader = NULL;
    uint64_t num_of_locs;
    int i, j;
    AnalysisData data;

    if (argc != NARGS) {
        printf("usage: %s <trace file name>\n", argv[ARG0]);
        return 0;
    }

    reader = OTF2_Reader_Open(argv[TRACENAME]);
    if (!reader) {
        fprintf(stderr, "failed to open reader\n");
        return 1;
    }

    data.reader = reader;
    OTF2_Reader_SetSerialCollectiveCallbacks(reader);
    OTF2_Reader_GetNumberOfLocations(reader, &num_of_locs);
    data.matrix_size = num_of_locs;

    data.comm_matrix = calloc(data.matrix_size * data.matrix_size,
                              sizeof(data.comm_matrix[0]));

    if (!data.comm_matrix) {
        fprintf(stderr, "memory allocation error\n");
        OTF2_Reader_Close(reader);
        return 1;
    }

    read_global_definitions(reader, &data);
    read_events(reader, &data);
    
    OTF2_Reader_Close(reader);

    /* communication matrix print */
    FILE *matrix = fopen("matrix", "w+");
    for (i = 0; i < data.matrix_size; ++i) {
        for (j = 0; j < data.matrix_size; ++j) {
            fprintf(matrix ,"%f ", data.comm_matrix[i * data.matrix_size + j]);
        }
        fprintf(matrix ,"\n");
    }
    free(data.comm_matrix);
    return 0;
}
